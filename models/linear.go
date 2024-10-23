package models

import (
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/aouyang1/go-forecaster/array"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

var (
	ErrNoOptions          = errors.New("no initialized model options")
	ErrTargetLenMismatch  = errors.New("target length does not match target rows")
	ErrNoTrainingArray    = errors.New("no training array")
	ErrNoTargetArray      = errors.New("no target array")
	ErrNoDesignMatrix     = errors.New("no design matrix for inference")
	ErrFeatureLenMismatch = errors.New("number of features does not match number of model coefficients")
)

type OLSOptions struct {
	FitIntercept bool
}

func NewDefaulOLSOptions() *OLSOptions {
	return &OLSOptions{
		FitIntercept: true,
	}
}

// OLSRegression computes ordinary least squares using QR factorization
type OLSRegression struct {
	opt       *OLSOptions
	coef      []float64
	intercept float64
}

func NewOLSRegression(opt *OLSOptions) (*OLSRegression, error) {
	if opt == nil {
		opt = NewDefaulOLSOptions()
	}
	return &OLSRegression{
		opt: opt,
	}, nil
}

func (o *OLSRegression) Fit(x, y *array.Array) error {
	if o.opt == nil {
		return ErrNoOptions
	}
	if x == nil {
		return ErrNoTrainingArray
	}
	if y == nil {
		return ErrNoTargetArray
	}
	m, n := x.Shape()

	ym, _ := y.Shape()
	if ym != m {
		return fmt.Errorf("training data has %d rows and target has %d row, %w", m, ym, ErrTargetLenMismatch)
	}

	if o.opt.FitIntercept {
		ones, err := array.Ones(m, 1)
		if err != nil {
			return err
		}
		x, err = array.Extend(ones, x)
		if err != nil {
			return err
		}
		m, n = x.Shape()
	}

	X := mat.NewDense(m, n, x.Flatten())
	Y := mat.NewDense(1, m, y.Flatten())

	qr := new(mat.QR)
	qr.Factorize(X)

	q := new(mat.Dense)
	r := new(mat.Dense)

	qr.QTo(q)
	qr.RTo(r)
	yq := new(mat.Dense)
	yq.Mul(Y, q)

	c := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		c[i] = yq.At(0, i)
		for j := i + 1; j < n; j++ {
			c[i] -= c[j] * r.At(i, j)
		}
		c[i] /= r.At(i, i)
	}

	if o.opt.FitIntercept {
		o.intercept = c[0]
		o.coef = c[1:]
	} else {
		o.coef = c
	}

	return nil
}

func (o *OLSRegression) Predict(x *array.Array) ([]float64, error) {
	if o.opt == nil {
		return nil, ErrNoOptions
	}
	if x == nil {
		return nil, ErrNoDesignMatrix
	}

	coef := o.coef
	if o.opt.FitIntercept {
		coef = append([]float64{o.intercept}, o.coef...)

		m, _ := x.Shape()
		ones, err := array.Ones(m, 1)
		if err != nil {
			return nil, err
		}
		x, err = array.Extend(ones, x)
		if err != nil {
			return nil, err
		}
	}
	n := len(coef)

	xT := x.T()
	xn, xm := xT.Shape()
	if xn != n {
		return nil, fmt.Errorf("got %d features in design matrix, but expected %d, %w", xn, n, ErrFeatureLenMismatch)
	}
	coefMx := mat.NewDense(1, n, coef)
	desMx := mat.NewDense(n, xm, xT.Flatten())

	var res mat.Dense
	res.Mul(coefMx, desMx)
	return res.RawRowView(0), nil
}

func (o *OLSRegression) Score(x, y *array.Array) (float64, error) {
	if o.opt == nil {
		return 0.0, ErrNoOptions
	}
	if x == nil {
		return 0.0, ErrNoDesignMatrix
	}
	if y == nil {
		return 0.0, ErrNoTargetArray
	}

	m, _ := x.Shape()

	ym, _ := y.Shape()
	if m != ym {
		return 0.0, fmt.Errorf("design matrix has %d rows and target has %d rows, %w", m, ym, ErrTargetLenMismatch)
	}

	res, err := o.Predict(x)
	if err != nil {
		return 0.0, err
	}

	return stat.RSquaredFrom(res, y.Flatten(), nil), nil
}

func (o *OLSRegression) Intercept() float64 {
	return o.intercept
}

func (o *OLSRegression) Coef() []float64 {
	c := make([]float64, len(o.coef))
	copy(c, o.coef)
	return c
}

var (
	ErrObsYSizeMismatch  = errors.New("observation and y matrix have different number of features")
	ErrWarmStartBetaSize = errors.New("warm start beta does not have the same dimensions as X")
)

// LassoOptions represents input options too run the Lasso Regression
type LassoOptions struct {
	WarmStartBeta []float64
	Lambda        float64
	Iterations    int
	Tolerance     float64
	FitIntercept  bool
}

// NewDefaultLassoOptions returns a default set of Lasso Regression options
func NewDefaultLassoOptions() *LassoOptions {
	return &LassoOptions{
		Lambda:        1.0,
		Iterations:    1000,
		Tolerance:     1e-4,
		WarmStartBeta: nil,
		FitIntercept:  true,
	}
}

// LassoRegression computes the lasso regression using coordinate descent. lambda = 0 converges to OLS
// The obs first dimension represents columns or features starting with the intercept.
type LassoRegression struct {
	opt *LassoOptions

	coef      []float64
	intercept float64
}

func NewLassoRegression(opt *LassoOptions) (*LassoRegression, error) {
	if opt == nil {
		opt = NewDefaultLassoOptions()
	}
	return &LassoRegression{
		opt: opt,
	}, nil
}

func (l *LassoRegression) Fit(x, y *array.Array) error {
	if l.opt == nil {
		return ErrNoOptions
	}
	if x == nil {
		return ErrNoTrainingArray
	}
	if y == nil {
		return ErrNoTargetArray
	}

	m, n := x.Shape()

	ym, _ := y.Shape()
	if ym != m {
		return fmt.Errorf("training data has %d rows and target has %d row, %w", m, ym, ErrTargetLenMismatch)
	}

	if l.opt.FitIntercept {
		ones, err := array.Ones(m, 1)
		if err != nil {
			return err
		}
		x, err = array.Extend(ones, x)
		if err != nil {
			return err
		}
		m, n = x.Shape()
	}

	if l.opt.WarmStartBeta != nil && len(l.opt.WarmStartBeta) != n {
		return fmt.Errorf("warm start beta has %d features instead of %d, %w", len(l.opt.WarmStartBeta), n, ErrWarmStartBetaSize)
	}

	// tracks current betas
	beta := make([]float64, n)
	if l.opt.WarmStartBeta != nil {
		copy(beta, l.opt.WarmStartBeta)
	}

	// precompute the per feature dot product
	xdot := make([]float64, n)
	for i := 0; i < n; i++ {
		xi, err := x.GetCol(i)
		if err != nil {
			return fmt.Errorf("attempting to access col index %d with %d columns during preprocessing, %w", i, n, err)
		}
		xdot[i] = floats.Dot(xi, xi)
	}

	// tracks the per coordinate residual
	residual := make([]float64, m)
	betaX := make([]float64, m)
	betaXDelta := make([]float64, m)

	yArr, err := y.GetCol(0)
	if err != nil {
		return fmt.Errorf("attempting to access column of target array, %w", err)
	}
	for i := 0; i < l.opt.Iterations; i++ {
		maxCoef := 0.0
		maxUpdate := 0.0
		betaDiff := 0.0

		// loop through all features and minimize loss function
		for j := 0; j < n; j++ {
			betaCurr := beta[j]
			if i != 0 {
				if betaCurr == 0 {
					continue
				}
			}

			floats.Add(betaX, betaXDelta)
			floats.SubTo(residual, yArr, betaX)

			obsCol, err := x.GetCol(j)
			if err != nil {
				return fmt.Errorf("attempting to access col index %d with %d columns during fit, %w", i, n, err)
			}
			num := floats.Dot(obsCol, residual)
			betaNext := num/xdot[j] + betaCurr

			gamma := l.opt.Lambda / xdot[j]
			betaNext = SoftThreshold(betaNext, gamma)

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
	} else {
		l.coef = beta
	}

	return nil
}

func (l *LassoRegression) Predict(x *array.Array) ([]float64, error) {
	if l.opt == nil {
		return nil, ErrNoOptions
	}
	if x == nil {
		return nil, ErrNoDesignMatrix
	}

	coef := l.coef
	if l.opt.FitIntercept {
		coef = append([]float64{l.intercept}, l.coef...)

		m, _ := x.Shape()
		ones, err := array.Ones(m, 1)
		if err != nil {
			return nil, err
		}
		x, err = array.Extend(ones, x)
		if err != nil {
			return nil, err
		}
	}
	n := len(coef)

	xT := x.T()
	xn, xm := xT.Shape()
	if xn != n {
		return nil, fmt.Errorf("got %d features in design matrix, but expected %d, %w", xn, n, ErrFeatureLenMismatch)
	}
	coefMx := mat.NewDense(1, n, coef)
	desMx := mat.NewDense(n, xm, xT.Flatten())

	var res mat.Dense
	res.Mul(coefMx, desMx)
	return res.RawRowView(0), nil
}

func (l *LassoRegression) Intercept() float64 {
	return l.intercept
}

func (l *LassoRegression) Coef() []float64 {
	return l.coef
}

func (l *LassoRegression) Score(x, y *array.Array) (float64, error) {
	if l.opt == nil {
		return 0.0, ErrNoOptions
	}
	if x == nil {
		return 0.0, ErrNoDesignMatrix
	}
	if y == nil {
		return 0.0, ErrNoTargetArray
	}

	m, _ := x.Shape()

	ym, _ := y.Shape()
	if m != ym {
		return 0.0, fmt.Errorf("design matrix has %d rows and target has %d rows, %w", m, ym, ErrTargetLenMismatch)
	}

	res, err := l.Predict(x)
	if err != nil {
		return 0.0, err
	}

	return stat.RSquaredFrom(res, y.Flatten(), nil), nil
}

// SoftThreshold returns 0 if the value is less than or equal to the gamma input
func SoftThreshold(x, gamma float64) float64 {
	res := math.Max(0, math.Abs(x)-gamma)
	if math.Signbit(x) {
		return -res
	}
	return res
}

var (
	ErrInsufficientSamples       = errors.New("insufficient samples for the determined folds")
	ErrInconsistentSampleLengths = errors.New("features or targets do not have the same number of samples")
)

type FoldDataset struct {
	TrainX []time.Time
	TrainY []float64

	TestX []time.Time
	TestY []float64
}

func TimeSeriesCVSplit(t []time.Time, y []float64, nFold int) ([]FoldDataset, error) {
	nSamples := len(t)

	if len(y) != nSamples {
		return nil, ErrInconsistentSampleLengths
	}

	foldSamp := nSamples / (nFold + 1)
	if foldSamp == 0 {
		return nil, ErrInsufficientSamples
	}

	folds := make([]FoldDataset, nFold)
	for i := 0; i < nFold; i++ {
		trainX := t[:(i+1)*foldSamp]
		trainY := y[:(i+1)*foldSamp]

		testX := t[(i+1)*foldSamp : (i+2)*foldSamp]
		testY := y[(i+1)*foldSamp : (i+2)*foldSamp]
		si := FoldDataset{
			TrainX: trainX,
			TrainY: trainY,
			TestX:  testX,
			TestY:  testY,
		}
		folds[i] = si
	}
	return folds, nil
}
