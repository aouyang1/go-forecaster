// Package models is a collection of linear regression fitting implementations to be used in the
// forecaster
package models

import (
	"gonum.org/v1/gonum/mat"
)

type Model interface {
	Fit(x, y mat.Matrix) error
	Predict(x mat.Matrix) ([]float64, error)
	Score(x, y mat.Matrix) (float64, error)
	Intercept() float64
	Coef() []float64
}
