package models

import (
	"errors"
	"fmt"
	"math"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

var ErrWarmStartBetaSize = errors.New("warm start beta does not have the same number of coefficients as training features")

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
type LassoRegression struct {
	opt *LassoOptions

	coef      []float64
	intercept float64
}

// NewLassoRegression initializes a Lasso model ready for fitting
func NewLassoRegression(opt *LassoOptions) (*LassoRegression, error) {
	if opt == nil {
		opt = NewDefaultLassoOptions()
	}
	return &LassoRegression{
		opt: opt,
	}, nil
}

// Fit the model according to the given training data
func (l *LassoRegression) Fit(x, y mat.Matrix) error {
	if l.opt == nil {
		return ErrNoOptions
	}
	if x == nil {
		return ErrNoTrainingMatrix
	}
	if y == nil {
		return ErrNoTargetMatrix
	}

	m, n := x.Dims()

	ym, _ := y.Dims()
	if ym != m {
		return fmt.Errorf("training data has %d rows and target has %d row, %w", m, ym, ErrTargetLenMismatch)
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
		return fmt.Errorf("warm start beta has %d features instead of %d, %w", len(l.opt.WarmStartBeta), n, ErrWarmStartBetaSize)
	}

	// tracks current betas
	beta := make([]float64, n)
	if l.opt.WarmStartBeta != nil {
		copy(beta, l.opt.WarmStartBeta)
	}

	xcols := make([][]float64, n)

	// precompute the per feature dot product
	xdot := make([]float64, n)
	for i := 0; i < n; i++ {
		xi := mat.Col(nil, i, x)
		xdot[i] = floats.Dot(xi, xi)
		xcols[i] = xi
	}

	// tracks the per coordinate residual
	residual := make([]float64, m)

	// tracks the current beta * x by adding the deltas on each beta iteration
	betaX := make([]float64, m)

	// tracks the delta of the beta * x of each iteration by computing the next beta
	// multiplied by the feature observations of that beta. will be added to betaX on
	// the next beta iteration
	betaXDelta := make([]float64, m)

	yArr := mat.Col(nil, 0, y)
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

			obsCol := xcols[j]
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

	return stat.RSquaredFrom(res, ySlice, nil), nil
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
