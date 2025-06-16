package models

import (
	"errors"
	"fmt"
	"log/slog"
	"math"
	"sync"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

const (
	DefaultLambda     = 1.0
	DefaultIterations = 1000
	DefaultTolerance  = 1e-4
)

var (
	ErrNegativeLambda     = errors.New("negative lambda")
	ErrNegativeIterations = errors.New("negative iterations")
	ErrNegativeTolerance  = errors.New("negative tolerance")
	ErrWarmStartBetaSize  = errors.New("warm start beta does not have the same number of coefficients as training features")
	ErrNoLambdas          = errors.New("no lambdas provided to fit with")
)

// LassoOptions represents input options to run the Lasso Regression
type LassoOptions struct {
	// WarmStartBeta is used to prime the coordinate descent to reduce the training time if a previous
	// fit has been performed.
	WarmStartBeta []float64

	// Lambda represents the L1 multiplier, controlling the regularization. Must be a non-negative. 0.0 results in converging
	// to Ordinary Least Squares (OLS).
	Lambda float64

	// Iterations is the maximum number of times the fit loops through training all coefficients.
	Iterations int

	// Tolerance is the smallest coefficient channge on each iteration to determine when to stop iterating.
	Tolerance float64

	// FitIntercept adds a constant 1.0 feature as the first column if set to true
	FitIntercept bool
}

// Validate runs basic validation on Lasso options
func (l *LassoOptions) Validate() (*LassoOptions, error) {
	if l == nil {
		l = NewDefaultLassoOptions()
	}

	if l.Lambda < 0 {
		return nil, ErrNegativeLambda
	}
	if l.Iterations < 0 {
		return nil, ErrNegativeIterations
	}
	if l.Tolerance < 0 {
		return nil, ErrNegativeTolerance
	}
	return l, nil
}

// NewDefaultLassoOptions returns a default set of Lasso Regression options
func NewDefaultLassoOptions() *LassoOptions {
	return &LassoOptions{
		Lambda:        DefaultLambda,
		Iterations:    DefaultIterations,
		Tolerance:     DefaultTolerance,
		WarmStartBeta: nil,
		FitIntercept:  true,
	}
}

// LassoRegression computes the lasso regression using coordinate descent. lambda = 0 converges to OLS
type LassoRegression struct {
	opt *LassoOptions

	// serve as precomputed data structures to reduce memory allocations
	xcols [][]float64
	xdot  []float64
	gamma []float64
	yArr  []float64

	coef      []float64
	intercept float64
}

// NewLassoRegression initializes a Lasso model ready for fitting
func NewLassoRegression(opt *LassoOptions) (*LassoRegression, error) {
	opt, err := opt.Validate()
	if err != nil {
		return nil, err
	}
	return &LassoRegression{
		opt: opt,
	}, nil
}

// Fit the model according to the given training data
func (l *LassoRegression) Fit(x, y mat.Matrix) error {
	x, y, err := l.fitValidate(x, y)
	if err != nil {
		return err
	}
	m, n := x.Dims()

	// tracks current betas
	beta := make([]float64, n)
	if l.opt.WarmStartBeta != nil {
		copy(beta, l.opt.WarmStartBeta)
	}

	// precompute data structures if not previously populated. This is generally only done
	// by the auto lasso regression
	l.precompute(n, m, x, y)

	// tracks the per coordinate residual
	residual := make([]float64, m)

	// tracks the current beta * x by adding the deltas on each beta iteration
	betaX := make([]float64, m)

	// tracks the delta of the beta * x of each iteration by computing the next beta
	// multiplied by the feature observations of that beta. will be added to betaX on
	// the next beta iteration
	betaXDelta := make([]float64, m)

	for i := 0; i < l.opt.Iterations; i++ {
		maxCoef := 0.0
		maxUpdate := 0.0
		betaDiff := 0.0

		// loop through all features and minimize loss function
		for j := 0; j < n; j++ {
			betaCurr := beta[j]
			if i != 0 && betaCurr == 0 {
				continue
			}

			floats.Add(betaX, betaXDelta)
			floats.SubTo(residual, l.yArr, betaX)

			obsCol := l.xcols[j]
			num := floats.Dot(obsCol, residual)
			betaNext := num/l.xdot[j] + betaCurr

			betaNext = SoftThreshold(betaNext, l.gamma[j])

			maxCoef = math.Max(maxCoef, betaNext)
			maxUpdate = math.Max(maxUpdate, math.Abs(betaNext-betaCurr))
			betaDiff = betaNext - betaCurr
			floats.ScaleTo(betaXDelta, betaDiff, obsCol)
			beta[j] = betaNext
		}

		// break early if we've achieved the desired tolerance
		if maxUpdate < l.opt.Tolerance*maxCoef {
			break
		}
	}

	if l.opt.FitIntercept {
		l.intercept = beta[0]
		l.coef = beta[1:]
		return nil
	}
	l.coef = beta
	return nil
}

func (l *LassoRegression) fitValidate(x, y mat.Matrix) (mat.Matrix, mat.Matrix, error) {
	if l.opt == nil {
		return nil, nil, ErrNoOptions
	}
	if x == nil {
		return nil, nil, ErrNoTrainingMatrix
	}
	if y == nil {
		return nil, nil, ErrNoTargetMatrix
	}

	m, n := x.Dims()

	ym, _ := y.Dims()
	if ym != m {
		return nil, nil, fmt.Errorf("training data has %d rows and target has %d row, %w", m, ym, ErrTargetLenMismatch)
	}

	if l.opt.FitIntercept {
		ones := make([]float64, m)
		floats.AddConst(1.0, ones)
		onesMx := mat.NewDense(1, m, ones)
		xT := x.T()

		var xWithOnes mat.Dense
		xWithOnes.Stack(onesMx, xT)
		x = xWithOnes.T()
		_, n = x.Dims()
	}

	if l.opt.WarmStartBeta != nil && len(l.opt.WarmStartBeta) != n {
		return nil, nil, fmt.Errorf("warm start beta has %d features instead of %d, %w", len(l.opt.WarmStartBeta), n, ErrWarmStartBetaSize)
	}
	return x, y, nil
}

func (l *LassoRegression) precompute(n, m int, x, y mat.Matrix) {
	if len(l.xdot) != 0 || len(l.xcols) != 0 || len(l.gamma) != 0 || len(l.yArr) != 0 {
		return
	}
	l.xcols = make([][]float64, n)
	for i := 0; i < n; i++ {
		l.xcols[i] = make([]float64, m)
	}

	// precompute the per feature dot product
	l.xdot = make([]float64, n)
	l.gamma = make([]float64, n)
	for i := 0; i < n; i++ {
		xi := mat.Col(nil, i, x)
		if len(xi) < m {
			xi = append(xi, make([]float64, m-len(xi))...)
		}
		l.xcols[i] = xi
		l.xdot[i] = floats.Dot(xi, xi)
		l.gamma[i] = l.opt.Lambda / l.xdot[i]
	}

	l.yArr = mat.Col(nil, 0, y)
	if len(l.yArr) < m {
		l.yArr = append(l.yArr, make([]float64, m-len(l.yArr))...)
	}
}

// Predict using the Lasso model
func (l *LassoRegression) Predict(x mat.Matrix) ([]float64, error) {
	if l.opt == nil {
		return nil, ErrNoOptions
	}
	if x == nil {
		return nil, ErrNoDesignMatrix
	}

	coef := l.coef
	if l.opt.FitIntercept {
		coef = append([]float64{l.intercept}, l.coef...)

		m, _ := x.Dims()
		ones := make([]float64, m)
		floats.AddConst(1.0, ones)
		onesMx := mat.NewDense(1, m, ones)
		xT := x.T()

		var xWithOnes mat.Dense
		xWithOnes.Stack(onesMx, xT)
		x = xWithOnes.T()
	}
	n := len(coef)

	_, xn := x.Dims()
	if xn != n {
		return nil, fmt.Errorf("got %d features in design matrix, but expected %d, %w", xn, n, ErrFeatureLenMismatch)
	}

	xT := x.T()
	coefMx := mat.NewDense(1, n, coef)

	var res mat.Dense
	res.Mul(coefMx, xT)
	return res.RawRowView(0), nil
}

// Score computes the coefficient of determination of the prediction
func (l *LassoRegression) Score(x, y mat.Matrix) (float64, error) {
	if l.opt == nil {
		return 0.0, ErrNoOptions
	}
	if x == nil {
		return 0.0, ErrNoDesignMatrix
	}
	if y == nil {
		return 0.0, ErrNoTargetMatrix
	}

	m, _ := x.Dims()

	ym, _ := y.Dims()
	if m != ym {
		return 0.0, fmt.Errorf("design matrix has %d rows and target has %d rows, %w", m, ym, ErrTargetLenMismatch)
	}

	res, err := l.Predict(x)
	if err != nil {
		return 0.0, err
	}

	ySlice := mat.Col(nil, 0, y)

	score := stat.RSquaredFrom(res, ySlice, nil)
	if math.IsNaN(score) {
		score = 1.0
	}

	return score, nil
}

// Intercept returns the computed intercept if FitIntercept is set to true. Defaults to 0.0 if not set.
func (l *LassoRegression) Intercept() float64 {
	return l.intercept
}

// Coef returns a slice of the trained coefficients in the same order of the training feature Matrix by column.
func (l *LassoRegression) Coef() []float64 {
	return l.coef
}

// SoftThreshold returns 0.0 if the value is less than or equal to the gamma input
func SoftThreshold(x, gamma float64) float64 {
	res := math.Max(0, math.Abs(x)-gamma)
	if math.Signbit(x) {
		return -res
	}
	return res
}

// LassoAutoOptions represents input options to run the Lasso Regression with optimal regularization parameter lambda
type LassoAutoOptions struct {
	// Lambda represents the L1 multiplier, controlling the regularization. Must be a non-negative. 0.0 results in converging
	// to Ordinary Least Squares (OLS).
	Lambdas []float64

	// Iterations is the maximum number of times the fit loops through training all coefficients.
	Iterations int

	// Tolerance is the smallest coefficient channge on each iteration to determine when to stop iterating.
	Tolerance float64

	// FitIntercept adds a constant 1.0 feature as the first column if set to true
	FitIntercept bool

	// Parallelization sets how many fits to run in parallel. More will increase memory and compute usage.
	Parallelization int
}

// Validate runs basic validation on Lasso Auto options
func (l *LassoAutoOptions) Validate() (*LassoAutoOptions, error) {
	if l == nil {
		l = NewDefaultLassoAutoOptions()
	}

	if len(l.Lambdas) == 0 {
		return nil, ErrNoLambdas
	}

	for _, lambda := range l.Lambdas {
		if lambda < 0.0 {
			return nil, ErrNegativeLambda
		}
	}

	if l.Iterations < 0 {
		return nil, ErrNegativeIterations
	}
	if l.Tolerance < 0 {
		return nil, ErrNegativeTolerance
	}
	if l.Parallelization == 0 || l.Parallelization > len(l.Lambdas) {
		l.Parallelization = len(l.Lambdas)
	}
	return l, nil
}

// NewDefaultLassoAutoOptions returns a default set of Lasso Auto Regression options
func NewDefaultLassoAutoOptions() *LassoAutoOptions {
	return &LassoAutoOptions{
		Lambdas:         []float64{DefaultLambda},
		Iterations:      DefaultIterations,
		Tolerance:       DefaultTolerance,
		FitIntercept:    true,
		Parallelization: 1,
	}
}

// LassoAutoRegression computes the lasso regression using coordinate descent. lambda is derived by finding the optimal
// regularization parameter
type LassoAutoRegression struct {
	opt *LassoAutoOptions

	// serve as precomputed data structures to reduce memory allocations
	xcols [][]float64
	xdot  []float64
	yArr  []float64

	scoreMu   sync.Mutex
	bestScore float64
	bestModel *LassoRegression
}

// NewLassoAutoRegression initializes a Lasso model ready for fitting using automated lambad parameter selection
func NewLassoAutoRegression(opt *LassoAutoOptions) (*LassoAutoRegression, error) {
	opt, err := opt.Validate()
	if err != nil {
		return nil, err
	}

	return &LassoAutoRegression{
		opt:       opt,
		bestScore: math.Inf(-1),
	}, nil
}

// Fit the model according to the given training data
func (l *LassoAutoRegression) Fit(x, y mat.Matrix) error {
	x, y, err := l.fitValidate(x, y)
	if err != nil {
		return err
	}
	m, n := x.Dims()

	lassoOpts := make([]*LassoOptions, 0, len(l.opt.Lambdas))
	for _, lambda := range l.opt.Lambdas {
		singleOpt := &LassoOptions{
			Lambda:       lambda,
			Iterations:   l.opt.Iterations,
			Tolerance:    l.opt.Tolerance,
			FitIntercept: false, // taken care of ahead of time
		}

		lassoOpts = append(lassoOpts, singleOpt)
	}

	l.precompute(n, m, x, y)

	sem := make(chan struct{}, l.opt.Parallelization)
	var wg sync.WaitGroup
	for _, lambda := range l.opt.Lambdas {
		sem <- struct{}{}
		wg.Add(1)

		go l.runLasso(lambda, x, y, &wg, sem)
	}
	wg.Wait()

	return nil
}

func (l *LassoAutoRegression) fitValidate(x, y mat.Matrix) (mat.Matrix, mat.Matrix, error) {
	if l.opt == nil {
		return nil, nil, ErrNoOptions
	}
	if x == nil {
		return nil, nil, ErrNoTrainingMatrix
	}
	if y == nil {
		return nil, nil, ErrNoTargetMatrix
	}

	m, _ := x.Dims()

	ym, _ := y.Dims()
	if ym != m {
		return nil, nil, fmt.Errorf("training data has %d rows and target has %d row, %w", m, ym, ErrTargetLenMismatch)
	}

	if l.opt.FitIntercept {
		ones := make([]float64, m)
		floats.AddConst(1.0, ones)
		onesMx := mat.NewDense(1, m, ones)
		xT := x.T()

		var xWithOnes mat.Dense
		xWithOnes.Stack(onesMx, xT)
		x = xWithOnes.T()
	}

	return x, y, nil
}

func (l *LassoAutoRegression) precompute(n, m int, x, y mat.Matrix) {
	l.xcols = make([][]float64, n)
	for i := 0; i < n; i++ {
		l.xcols[i] = make([]float64, m)
	}

	// precompute the per feature dot product
	l.xdot = make([]float64, n)
	for i := 0; i < n; i++ {
		xi := mat.Col(nil, i, x)
		if len(xi) < m {
			xi = append(xi, make([]float64, m-len(xi))...)
		}
		l.xcols[i] = xi
		l.xdot[i] = floats.Dot(xi, xi)
	}

	l.yArr = mat.Col(nil, 0, y)
	if len(l.yArr) < m {
		l.yArr = append(l.yArr, make([]float64, m-len(l.yArr))...)
	}
}

func (l *LassoAutoRegression) runLasso(lambda float64, x, y mat.Matrix, wg *sync.WaitGroup, sem chan struct{}) {
	defer func() {
		wg.Done()
		<-sem
	}()
	_, n := x.Dims()

	opt := &LassoOptions{
		Lambda:       lambda,
		Iterations:   l.opt.Iterations,
		Tolerance:    l.opt.Tolerance,
		FitIntercept: false, // taken care of ahead of time
	}

	gamma := make([]float64, n)
	for i := 0; i < n; i++ {
		gamma[i] = lambda / l.xdot[i]
	}
	reg, err := NewLassoRegression(opt)
	if err != nil {
		slog.Error("unable to initialize lasso regression", "error", err.Error())
		return
	}
	reg.xcols = l.xcols
	reg.xdot = l.xdot
	reg.gamma = gamma
	reg.yArr = l.yArr

	if err := reg.Fit(x, y); err != nil {
		slog.Error("unable to fit lasso regression", "error", err.Error())
		return
	}

	score, err := reg.Score(x, y)
	if err != nil {
		slog.Error("unable to compute fit score for lasso regression", "error", err.Error())
		return
	}

	l.scoreMu.Lock()
	defer l.scoreMu.Unlock()
	if score > l.bestScore {
		l.bestScore = score
		l.bestModel = reg
	}
}

// Predict using the Lasso model
func (l *LassoAutoRegression) Predict(x mat.Matrix) ([]float64, error) {
	if l.bestModel == nil {
		return nil, ErrNoOptions
	}

	if l.opt.FitIntercept {
		m, _ := x.Dims()
		ones := make([]float64, m)
		floats.AddConst(1.0, ones)
		onesMx := mat.NewDense(1, m, ones)
		xT := x.T()

		var xWithOnes mat.Dense
		xWithOnes.Stack(onesMx, xT)
		x = xWithOnes.T()
	}

	return l.bestModel.Predict(x)
}

// Score computes the coefficient of determination of the prediction
func (l *LassoAutoRegression) Score(x, y mat.Matrix) (float64, error) {
	if l.bestModel == nil {
		return 0.0, ErrNoOptions
	}

	if l.opt.FitIntercept {
		m, _ := x.Dims()
		ones := make([]float64, m)
		floats.AddConst(1.0, ones)
		onesMx := mat.NewDense(1, m, ones)
		xT := x.T()

		var xWithOnes mat.Dense
		xWithOnes.Stack(onesMx, xT)
		x = xWithOnes.T()
	}

	return l.bestModel.Score(x, y)
}

// Intercept returns the computed intercept if FitIntercept is set to true. Defaults to 0.0 if not set.
func (l *LassoAutoRegression) Intercept() float64 {
	if l == nil || l.bestModel == nil {
		return 0.0
	}
	if l.opt.FitIntercept {
		return l.bestModel.Coef()[0]
	}
	return 0.0
}

// Coef returns a slice of the trained coefficients in the same order of the training feature Matrix by column.
func (l *LassoAutoRegression) Coef() []float64 {
	if l == nil || l.bestModel == nil {
		return nil
	}
	if l.opt.FitIntercept {
		return l.bestModel.Coef()[1:]
	}
	return l.bestModel.Coef()
}
