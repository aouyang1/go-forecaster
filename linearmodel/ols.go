package linearmodel

import (
	"fmt"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// OLSOptions represents input options to run the OLS Regression
type OLSOptions struct {
	// FitIntercept adds a constant 1.0 feature as the first column if set to true
	FitIntercept bool
}

// Validate runs basic validation on OLS options
func (o *OLSOptions) Validate() (*OLSOptions, error) {
	if o == nil {
		o = NewDefaultOLSOptions()
	}

	return o, nil
}

// NewDefaultOLSOptions returns a default set of OLS Regression options
func NewDefaultOLSOptions() *OLSOptions {
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

// NewOLSRegression initializes an ordinary least squares model ready for fitting
func NewOLSRegression(opt *OLSOptions) (*OLSRegression, error) {
	opt, err := opt.Validate()
	if err != nil {
		return nil, err
	}
	return &OLSRegression{
		opt: opt,
	}, nil
}

// Fit the model according to the given training data
func (o *OLSRegression) Fit(x, y mat.Matrix) error {
	if o.opt == nil {
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

	if o.opt.FitIntercept {
		ones := make([]float64, m)
		floats.AddConst(1.0, ones)
		onesMx := mat.NewDense(1, m, ones)
		xT := x.T()

		var xWithOnes mat.Dense
		xWithOnes.Stack(onesMx, xT)
		x = xWithOnes.T()
		_, n = x.Dims()
	}

	yT := y.T()

	qr := new(mat.QR)
	qr.Factorize(x)

	q := new(mat.Dense)
	r := new(mat.Dense)

	qr.QTo(q)
	qr.RTo(r)
	yq := new(mat.Dense)
	yq.Mul(yT, q)

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

// Predict using the OLS model
func (o *OLSRegression) Predict(x mat.Matrix) ([]float64, error) {
	if o.opt == nil {
		return nil, ErrNoOptions
	}
	if x == nil {
		return nil, ErrNoDesignMatrix
	}

	coef := o.coef
	if o.opt.FitIntercept {
		coef = append([]float64{o.intercept}, o.coef...)

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

	xT := x.T()
	xn, _ := xT.Dims()
	if xn != n {
		return nil, fmt.Errorf("got %d features in design matrix, but expected %d, %w", xn, n, ErrFeatureLenMismatch)
	}
	coefMx := mat.NewDense(1, n, coef)

	var res mat.Dense
	res.Mul(coefMx, xT)
	return res.RawRowView(0), nil
}

// Score computes the coefficient of determination of the prediction
func (o *OLSRegression) Score(x, y mat.Matrix) (float64, error) {
	if o.opt == nil {
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

	res, err := o.Predict(x)
	if err != nil {
		return 0.0, err
	}

	ySlice := mat.Col(nil, 0, y)

	return stat.RSquaredFrom(res, ySlice, nil), nil
}

// Intercept returns the computed intercept if FitIntercept is set to true. Defaults to 0.0 if not set.
func (o *OLSRegression) Intercept() float64 {
	return o.intercept
}

// Coef returns a slice of the trained coefficients in the same order of the training feature Matrix by column.
func (o *OLSRegression) Coef() []float64 {
	c := make([]float64, len(o.coef))
	copy(c, o.coef)
	return c
}
