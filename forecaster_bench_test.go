package forecaster

import (
	"os"
	"testing"
	"time"

	"github.com/goccy/go-json"
	"github.com/pkg/profile"
)

var benchPredictRes *Results

func BenchmarkTrainToModel(b *testing.B) {
	t, y, opt := setupWithOutliers()

	var f *Forecaster
	var err error

	b.ResetTimer()
	for b.Loop() {
		f, err = New(opt)
		if err != nil {
			panic(err)
		}

		if err := f.Fit(t, y); err != nil {
			panic(err)
		}
	}

	m, err := f.Model()
	if err != nil {
		panic(err)
	}

	bytes, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		panic(err)
	}

	if err := os.WriteFile("benchmark_model.json", bytes, 0o644); err != nil {
		panic(err)
	}
}

func BenchmarkPredictFromModel(b *testing.B) {
	bytes, err := os.ReadFile("benchmark_model.json")
	if err != nil {
		panic(err)
	}

	var model Model
	if err := json.Unmarshal(bytes, &model); err != nil {
		panic(err)
	}
	f, err := NewFromModel(model)
	if err != nil {
		panic(err)
	}

	input := make([]time.Time, 0, 2)
	ct := time.Now()
	for i := range cap(input) {
		input = append(input, ct.Add(time.Duration(i)*time.Minute))
	}
	b.ResetTimer()
	defer profile.Start(profile.CPUProfile, profile.ProfilePath(".")).Stop()
	for b.Loop() {
		benchPredictRes, err = f.Predict(input)
		if err != nil {
			panic(err)
		}
	}
}
