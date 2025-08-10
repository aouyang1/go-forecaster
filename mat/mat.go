// mat is an extension of the mat package in gonum
package mat

import (
	"errors"
	"fmt"

	"gonum.org/v1/gonum/mat"
)

var ErrColMismatch = errors.New("column size mismatch")

func NewDenseFromArray(x [][]float64) (*mat.Dense, error) {
	m := len(x)

	n := -1
	for i, row := range x {
		if n >= 0 && len(row) != n {
			return nil, fmt.Errorf("at row %d, %w", i, ErrColMismatch)
		}
		if n < 0 {
			n = len(row)
		}
	}
	if n < 0 {
		n = 0
	}

	// flatten to row order
	data := make([]float64, 0, m*n)
	for _, row := range x {
		data = append(data, row...)
	}
	return mat.NewDense(m, n, data), nil
}
