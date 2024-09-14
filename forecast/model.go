package forecast

import (
	"encoding/json"
	"errors"

	"github.com/aouyang1/go-forecaster/feature"
)

var ErrUnknownFeatureType = errors.New("unkown feature type")

type Model struct {
	Options *Options `json:"options"`
	Scores  *Scores  `json:"scores"`
	Weights Weights  `json:"weights"`
}

type Weights struct {
	Coef      []FeatureWeight `json:"coefficients"`
	Intercept float64         `json:"intercept"`
}

func (w *Weights) FeatureLabels() (*FeatureLabels, error) {
	labels := make([]feature.Feature, 0, len(w.Coef))
	for _, fw := range w.Coef {
		feat, err := fw.ToFeature()
		if err != nil {
			return nil, err
		}
		labels = append(labels, feat)
	}
	return NewFeatureLabels(labels), nil
}

func (w *Weights) Coefficients() []float64 {
	coef := make([]float64, 0, len(w.Coef))
	for _, fw := range w.Coef {
		coef = append(coef, fw.Value)
	}
	return coef
}

type FeatureWeight struct {
	Labels map[string]string   `json:"labels"`
	Type   feature.FeatureType `json:"type"`
	Value  float64             `json:"value"`
}

func (fw *FeatureWeight) ToFeature() (feature.Feature, error) {
	switch fw.Type {
	case feature.FeatureTypeChangepoint:
		bytes, err := json.Marshal(fw.Labels)
		if err != nil {
			return nil, err
		}
		feat := new(feature.Changepoint)
		if err := json.Unmarshal(bytes, feat); err != nil {
			return nil, err
		}
		return feat, nil

	case feature.FeatureTypeSeasonality:
		bytes, err := json.Marshal(fw.Labels)
		if err != nil {
			return nil, err
		}
		feat := new(feature.Seasonality)
		if err := json.Unmarshal(bytes, &feat); err != nil {
			return nil, err
		}
		return feat, nil
	}
	return nil, ErrUnknownFeatureType
}
