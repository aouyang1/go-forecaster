package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func Regression(obs, y mat.Matrix) []float64 {
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
	return c
}

func main() {
	data := make([]float64, 0, 15)
	for i := 0; i < cap(data); i++ {
		data = append(data, float64(i))
	}

	data2 := make([]float64, 0, 10)
	for i := 0; i < cap(data2); i++ {
		data2 = append(data2, float64(i))
	}

	a := mat.NewDense(3, 5, data)
	b := mat.NewDense(5, 2, data2)
	fmt.Println(a)
	fmt.Println(b)
	var c mat.Dense
	c.Mul(a, b)
	fmt.Println(c)
}
