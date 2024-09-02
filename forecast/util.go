package forecast

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"time"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

func featureMatrixChangepoints(t []time.Time, featureLabels []time.Time, features map[time.Time][]float64) *mat.Dense {
	m := len(t)
	n := len(features) + 1
	obs := make([]float64, m*n)

	featNum := 0
	for i := 0; i < m; i++ {
		idx := n * i
		obs[idx] = 1.0
	}
	featNum += 1

	for _, label := range featureLabels {
		feature := features[label]
		for i := 0; i < len(feature); i++ {
			idx := n*i + featNum
			obs[idx] = feature[i]
		}
		featNum += 1
	}
	return mat.NewDense(m, n, obs)
}

func featureMatrix(t []time.Time, featureLabels []string, features map[string][]float64) *mat.Dense {
	m := len(t)
	n := len(features) + 1
	obs := make([]float64, m*n)

	featNum := 0
	for i := 0; i < m; i++ {
		idx := n * i
		obs[idx] = 1.0
	}
	featNum += 1

	for _, label := range featureLabels {
		feature := features[label]
		for i := 0; i < len(feature); i++ {
			idx := n*i + featNum
			obs[idx] = feature[i]
		}
		featNum += 1
	}
	return mat.NewDense(m, n, obs)
}

func observationMatrix(y []float64) *mat.Dense {
	n := len(y)
	return mat.NewDense(1, n, y)
}

func generateTimeFeatures(t []time.Time, opt *Options) map[string][]float64 {
	if opt == nil {
		opt = NewDefaultOptions()
	}
	tFeat := make(map[string][]float64)
	if opt.DailyOrders > 0 {
		hod := make([]float64, len(t))
		for i, tPnt := range t {
			hour := float64(tPnt.Unix()) / 3600.0
			hod[i] = math.Mod(hour, 24.0)
		}
		tFeat["hod"] = hod
	}
	if opt.WeeklyOrders > 0 {
		dow := make([]float64, len(t))
		for i, tPnt := range t {
			day := float64(tPnt.Unix()) / 86400.0
			dow[i] = math.Mod(day, 7.0)
		}
		tFeat["dow"] = dow
	}
	return tFeat
}

func featureLabels(features map[string][]float64) []string {
	labels := make([]string, 0, len(features))
	for label := range features {
		labels = append(labels, label)
	}
	sort.Slice(
		labels,
		func(i, j int) bool {
			return labels[i] < labels[j]
		},
	)
	return labels
}

func generateFourierFeatures(tFeat map[string][]float64, opt *Options) (map[string][]float64, error) {
	if opt == nil {
		opt = NewDefaultOptions()
	}
	x := make(map[string][]float64)
	if opt.DailyOrders > 0 {
		var orders []int
		for i := 1; i <= opt.DailyOrders; i++ {
			orders = append(orders, i)
		}
		dailyFeatures, err := generateFourierOrders(tFeat, "hod", orders, 24.0)
		if err != nil {
			return nil, fmt.Errorf("%q not present in time features, %w", "hod", err)
		}
		for label, features := range dailyFeatures {
			x[label] = features
		}
	}

	if opt.WeeklyOrders > 0 {
		var orders []int
		for i := 1; i <= opt.WeeklyOrders; i++ {
			if i%7 == 0 && i/7 <= opt.DailyOrders {
				// colinear feature so skip
				continue
			}
			orders = append(orders, i)
		}
		weeklyFeatures, err := generateFourierOrders(tFeat, "dow", orders, 7.0)
		if err != nil {
			return nil, fmt.Errorf("%q not present in time features, %w", "dow", err)
		}
		for label, features := range weeklyFeatures {
			x[label] = features
		}
	}
	return x, nil
}

func generateFourierOrders(tFeatures map[string][]float64, col string, orders []int, period float64) (map[string][]float64, error) {
	tFeat, exists := tFeatures[col]
	if !exists {
		return nil, ErrUnknownTimeFeature
	}

	x := make(map[string][]float64)
	for _, order := range orders {
		sinFeat, cosFeat := generateFourierComponent(tFeat, order, period)
		sinFeatCol := fmt.Sprintf("%s_%02dsin", col, order)
		cosFeatCol := fmt.Sprintf("%s_%02dcos", col, order)
		x[sinFeatCol] = sinFeat
		x[cosFeatCol] = cosFeat
	}
	return x, nil
}

func generateFourierComponent(timeFeature []float64, order int, period float64) ([]float64, []float64) {
	omega := 2.0 * math.Pi * float64(order) / period
	sinFeat := make([]float64, len(timeFeature))
	cosFeat := make([]float64, len(timeFeature))
	for i, tFeat := range timeFeature {
		rad := omega * tFeat
		sinFeat[i] = math.Sin(rad)
		cosFeat[i] = math.Cos(rad)
	}
	return sinFeat, cosFeat
}

var (
	ErrFeatureLenMismatch = errors.New("some feature length is not consistent")
	ErrMinimumFeatures    = errors.New("need at least 2 features to coompute VIF")
	ErrFeatureLen         = errors.New("must have at least 2 points per feature")
)

func generateChangepointFeatures(t, chpts []time.Time) map[string][]float64 {
	var minTime, maxTime time.Time
	for _, tPnt := range t {
		if minTime.IsZero() || tPnt.Before(minTime) {
			minTime = tPnt
		}
		if maxTime.IsZero() || tPnt.After(maxTime) {
			maxTime = tPnt
		}
	}

	sort.Slice(
		chpts,
		func(i, j int) bool {
			return chpts[i].Before(chpts[j])
		},
	)
	chptStart := len(chpts)
	chptEnd := -1
	for i := 0; i < len(chpts); i++ {
		// haven't reached a changepoint in the time window
		if chpts[i].Before(minTime) {
			continue
		}
		if i < chptStart {
			chptStart = i
		}

		// reached end of time window so break
		if chpts[i].Equal(maxTime) || chpts[i].After(maxTime) {
			chptEnd = i
			break
		}
	}
	if chptEnd == -1 {
		chptEnd = len(chpts)
	}
	fChpts := chpts[chptStart:chptEnd]
	chptFeatures := make([][]float64, len(fChpts)*2)
	for i := 0; i < len(fChpts)*2; i++ {
		chpt := make([]float64, len(t))
		chptFeatures[i] = chpt
	}

	bias := 1.0
	var slope float64
	for i := 0; i < len(t); i++ {
		for j := 0; j < len(fChpts); j++ {
			var beforeNextChpt bool
			if j != len(fChpts)-1 {
				beforeNextChpt = t[i].Before(fChpts[j+1])
			} else {
				beforeNextChpt = true
			}
			if t[i].Equal(fChpts[j]) || (t[i].After(fChpts[j]) && beforeNextChpt) {
				slope = t[i].Sub(fChpts[j]).Seconds()
				chptFeatures[j*2][i] = bias
				chptFeatures[j*2+1][i] = slope
			}
		}
	}

	feat := make(map[string][]float64)
	for i := 0; i < len(fChpts); i++ {
		feat[fmt.Sprintf("chpnt_bias_%02d", i)] = chptFeatures[i*2]
		feat[fmt.Sprintf("chpnt_slope_%02d", i)] = chptFeatures[i*2+1]
	}
	return feat
}

func VarianceInflationFactor(features map[string][]float64) (map[string]float64, error) {
	if len(features) < 2 {
		return nil, ErrMinimumFeatures
	}
	n := len(features)
	var m int
	for _, feature := range features {
		if len(feature) < 2 {
			return nil, ErrFeatureLen
		}
		if m == 0 {
			m = len(feature)
			continue
		}
		if m != len(feature) {
			return nil, ErrFeatureLenMismatch
		}
	}

	vif := make(map[string]float64)
	x := mat.NewDense(m, n, nil)
	y := mat.NewDense(1, m, nil)

	ones := make([]float64, m)
	floats.AddConst(1.0, ones)
	x.SetCol(0, ones)

	w := make([]float64, n)
	weightsMx := mat.NewDense(1, n, nil)
	for label, labelFeature := range features {
		y.SetRow(0, labelFeature)
		c := 1
		for otherLabel, otherLabelFeature := range features {
			if otherLabel == label {
				continue
			}
			x.SetCol(c, otherLabelFeature)
			c++
		}
		intercept, weights := OLS(x, y)
		w[0] = intercept
		for i, weight := range weights {
			w[i+1] = weight
		}
		weightsMx.SetRow(0, w)

		var predictedMx mat.Dense
		predictedMx.Mul(weightsMx, x.T())
		predicted := mat.Row(nil, 0, &predictedMx)

		vif[label] = stat.RSquaredFrom(predicted, labelFeature, nil)
	}
	return vif, nil
}

// OLS computes ordinary least squares using QR factorization
func OLS(obs, y mat.Matrix) (float64, []float64) {
	_, n := obs.Dims()
	qr := new(mat.QR)
	qr.Factorize(obs)

	q := new(mat.Dense)
	r := new(mat.Dense)

	qr.QTo(q)
	qr.RTo(r)
	yq := new(mat.Dense)
	yq.Mul(y, q)

	c := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		c[i] = yq.At(0, i)
		for j := i + 1; j < n; j++ {
			c[i] -= c[j] * r.At(i, j)
		}
		c[i] /= r.At(i, i)
	}
	if len(c) == 0 {
		return math.NaN(), nil
	}
	if len(c) == 1 {
		return c[0], nil
	}
	return c[0], c[1:]
}

var (
	ErrObsYSizeMismatch  = errors.New("observation and y matrix have different number of features")
	ErrWarmStartBetaSize = errors.New("warm start beta does not have the same dimensions as X")
)

type LassoOptions struct {
	WarmStartBeta []float64
	Lambda        float64
	Iterations    int
	Tolerance     float64
}

func NewDefaultLassoOptions() *LassoOptions {
	return &LassoOptions{
		Lambda:        1.0,
		Iterations:    1000,
		Tolerance:     1e-4,
		WarmStartBeta: nil,
	}
}

// LassoRegression computes the lasso regression using coordinate descent. lambda = 0 converges to OLS
func LassoRegression(obs, y *mat.Dense, opt *LassoOptions) (float64, []float64, error) {
	if opt == nil {
		opt = NewDefaultLassoOptions()
	}

	m, n := obs.Dims()

	_, ym := y.Dims()
	if m != ym {
		return 0, nil, fmt.Errorf("observation matrix has %d observations and y matrix as %d observations, %w", m, ym, ErrObsYSizeMismatch)
	}
	if opt.WarmStartBeta != nil && len(opt.WarmStartBeta) != n {
		return 0, nil, fmt.Errorf("warm start beta has %d features instead of %d, %w", len(opt.WarmStartBeta), n, ErrWarmStartBetaSize)
	}

	// tracks current betas
	beta := mat.NewDense(1, n, opt.WarmStartBeta)

	// precompute the per feature dot product
	xdot := make([]float64, n)
	for i := 0; i < n; i++ {
		xi := obs.ColView(i)
		xdot[i] = mat.Dot(xi, xi)
	}

	// tracks the per coordinate residual
	residual := mat.NewDense(1, m, nil)

	for i := 0; i < opt.Iterations; i++ {
		maxCoef := 0.0
		maxUpdate := 0.0

		// loop through all features and minimize loss function
		for j := 0; j < n; j++ {
			residual.Mul(beta, obs.T())
			residual.Scale(-1, residual)

			residual.Add(y, residual)

			num := mat.Dot(obs.ColView(j), residual.RowView(0))
			betaCurr := beta.At(0, j)
			betaNext := num/xdot[j] + betaCurr

			gamma := opt.Lambda / xdot[j]
			betaNext = SoftThreshold(betaNext, gamma)

			maxCoef = math.Max(maxCoef, betaNext)
			maxUpdate = math.Max(maxUpdate, math.Abs(betaNext-betaCurr))
			beta.Set(0, j, betaNext)
		}

		// break early if we've achieved the desired tolerance
		if maxUpdate < opt.Tolerance*maxCoef {
			break
		}
	}

	c := beta.RawRowView(0)
	if len(c) == 0 {
		return math.NaN(), nil, nil
	}
	if len(c) == 1 {
		return c[0], nil, nil
	}
	return c[0], c[1:], nil
}

func SoftThreshold(x, gamma float64) float64 {
	res := math.Max(0, math.Abs(x)-gamma)
	if math.Signbit(x) {
		return -res
	}
	return res
}
