package array

import (
	"errors"
	"fmt"

	"gonum.org/v1/gonum/floats"
)

var (
	ErrNegativeDim        = errors.New("negative dimensions not allowed")
	ErrColMismatch        = errors.New("column size mismatch")
	ErrRowMismatch        = errors.New("row size mismatch")
	ErrUninitializedArray = errors.New("uninitialized array")
	ErrRowOutOfBounds     = errors.New("row is out of bounds")
	ErrColOutOfBounds     = errors.New("column is out of bounds")
)

// Array contains a 2D slice of data stored in column major order where the
// first slice in the stored slice is the first column of the dataset.
// e.g. [][]float64{{1.0, 2.0}, {1.0, 3.0}, {1.0, 4.0}} would be stored like so,
// {1.0, 1.0, 1.0, 2.0, 3.0, 4.0}.
type Array struct {
	arr []float64
	m   int
	n   int
}

func New1D(x []float64) *Array {
	a := new(Array)
	m := len(x)

	xArr := make([]float64, m)
	copy(xArr, x)

	a.arr = xArr
	a.m = m
	a.n = 1
	return a
}

func New2D(x [][]float64) (*Array, error) {
	a := new(Array)
	m, n, err := a.derive2DShape(x)
	if err != nil {
		return nil, err
	}

	xArr := make([]float64, m*n)
	for i, row := range x {
		for j, val := range row {
			xArr[j*m+i%m] = val
		}
	}

	a.arr = xArr
	a.m = m
	a.n = n
	return a, nil
}

func Zeros(m, n int) (*Array, error) {
	if m < 0 || n < 0 {
		return nil, ErrNegativeDim
	}
	ones := make([][]float64, m)
	for i := 0; i < m; i++ {
		ones[i] = make([]float64, n)
	}
	return New2D(ones)
}

func Ones(m, n int) (*Array, error) {
	res, err := Zeros(m, n)
	if err != nil {
		return nil, err
	}
	floats.AddConst(1.0, res.arr)
	return res, nil
}

func (a *Array) derive2DShape(x [][]float64) (int, int, error) {
	m := len(x)

	n := -1
	for i, row := range x {
		if n >= 0 && len(row) != n {
			return 0, 0, fmt.Errorf("at row %d, %w", i, ErrColMismatch)
		}
		if n < 0 {
			n = len(row)
		}
	}
	if n < 0 {
		n = 0
	}
	return m, n, nil
}

func (a *Array) Shape() (int, int) {
	return a.m, a.n
}

func (a *Array) Size() int {
	return len(a.arr)
}

// Get retrieves a single value in the array at a specific row and column
func (a *Array) Get(r, c int) (float64, error) {
	m, n := a.Shape()
	if r < 0 || r >= m {
		return 0.0, ErrRowOutOfBounds
	}
	if c < 0 || c >= n {
		return 0.0, ErrColOutOfBounds
	}

	idx := (r % m) + c*m
	return a.arr[idx], nil
}

// GetCol returns a slice view of the specified column
func (a *Array) GetCol(c int) ([]float64, error) {
	m, n := a.Shape()
	if c < 0 || c >= n {
		return nil, ErrColOutOfBounds
	}

	return a.arr[c*m : (c+1)*m], nil
}

// GetRow returns a slice view of the specified row
func (a *Array) GetRow(r int) ([]float64, error) {
	m, n := a.Shape()
	if r < 0 || r >= m {
		return nil, ErrRowOutOfBounds
	}

	res := make([]float64, 0, n)
	for c := 0; c < n; c++ {
		res = append(res, a.arr[c*m+r])
	}
	return res, nil
}

func (a *Array) Flatten() []float64 {
	m, n := a.Shape()
	res := make([]float64, a.Size())
	for i := 0; i < a.Size(); i++ {
		res[(i%m)*n+i/m] = a.arr[i]
	}
	return res
}

func (a *Array) ToSlice() [][]float64 {
	m, n := a.Shape()
	res := make([][]float64, m)
	for i := 0; i < m; i++ {
		res[i] = make([]float64, n)
	}
	for i := 0; i < a.Size(); i++ {
		row := i % m
		col := i / m
		res[row][col] = a.arr[i]
	}
	return res
}

func (a *Array) T() *Array {
	m, n := a.Shape()
	arr := make([]float64, a.Size())
	for i := 0; i < a.Size(); i++ {
		row := i % m
		col := i / m

		swapRow := col
		swapCol := row
		swapIdx := swapRow%n + swapCol*n

		arr[swapIdx] = a.arr[i]
	}
	return &Array{
		arr: arr,
		m:   n,
		n:   m,
	}
}

// Append adds rows to the first array from the second and returns a new array
func Append(a, b *Array) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("first array argument, %w", ErrUninitializedArray)
	}
	if b == nil {
		return nil, fmt.Errorf("second array argument, %w", ErrUninitializedArray)
	}
	aM, aN := a.Shape()
	bM, bN := b.Shape()
	if aN != bN {
		return nil, fmt.Errorf("first array with %d columns, and second array with %d columns, %w", aN, bN, ErrColMismatch)
	}

	m := aM + bM
	size := a.Size() + b.Size()
	arr := make([]float64, size)
	for i := 0; i < size; i++ {
		row := (i % m)
		col := i / m
		if row < aM {
			arr[i] = a.arr[row+col*aM]
		} else {
			arr[i] = b.arr[row-aM+col*bM]
		}
	}
	res := &Array{
		arr: arr,
		m:   aM + bM,
		n:   aN,
	}
	return res, nil
}

// Extend expands the first array adding more columns with the second and return
// a new array
func Extend(a, b *Array) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("first array argument, %w", ErrUninitializedArray)
	}
	if b == nil {
		return nil, fmt.Errorf("second array argument, %w", ErrUninitializedArray)
	}
	aM, aN := a.Shape()
	bM, bN := b.Shape()
	if aM != bM {
		return nil, fmt.Errorf("first array with %d rows, and second array with %d rows, %w", aM, bM, ErrRowMismatch)
	}

	arr := make([]float64, 0, a.Size()+b.Size())
	arr = append(arr, a.arr...)
	arr = append(arr, b.arr...)

	res := &Array{
		arr: arr,
		m:   aM,
		n:   aN + bN,
	}
	return res, nil
}
