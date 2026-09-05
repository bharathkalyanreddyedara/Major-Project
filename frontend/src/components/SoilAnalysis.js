import React, { useState } from 'react';

const SoilAnalysis = ({
  soilData,
  setSoilData,
  cnnResult,
  setCnnResult,
  onAnalyzeImage,
  onGetRecommendations,
  loading,
  imageLoading
}) => {
  const [preview, setPreview] = useState(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setPreview(URL.createObjectURL(file));
      onAnalyzeImage(file);
    }
  };

  const handleManualChange = (e) => {
    const { name, value } = e.target;
    setSoilData(prev => ({
      ...prev,
      [name]: parseFloat(value) || value
    }));
  };

  return (
    <div className="grid-2">
      {/* 1. Multimodal Soil Vision (CNN / Ensemble AI) */}
      <div className="card">
        <h3 className="card-title">📷 1. Live Soil Image Analysis</h3>
        <p className="card-subtitle">
          Upload any soil photo. The AI vision model extracts 133 multi-color and spatial texture features to classify soil type in real time.
        </p>

        <label className="dropzone" style={{ border: preview ? '2px solid #16a34a' : '2px dashed #cbd5e1' }}>
          <input
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            style={{ display: 'none' }}
          />
          <div style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}>🌱</div>
          <div style={{ fontWeight: '700', color: '#1e293b' }}>
            {preview ? 'Click to Change Soil Image' : 'Click or Drag & Drop Soil Photo'}
          </div>
          <div style={{ fontSize: '0.8rem', color: '#64748b' }}>Supports JPG, PNG, WEBP formats</div>
        </label>

        {preview && (
          <div style={{ textAlign: 'center', marginTop: '1rem' }}>
            <img src={preview} alt="Soil Preview" className="preview-image" style={{ maxHeight: '180px' }} />
          </div>
        )}

        {imageLoading && (
          <div style={{ padding: '1rem', textAlign: 'center', color: '#15803d', fontStyle: 'italic' }}>
            ⏳ Extracting 133 multi-channel features & evaluating ensemble model...
          </div>
        )}

        {cnnResult && (
          cnnResult.is_valid_soil === false ? (
            <div style={{ marginTop: '1.25rem', padding: '1.25rem', background: '#fef2f2', borderRadius: '12px', border: '1.5px solid #fca5a5' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.5rem' }}>
                <span style={{ fontSize: '1.4rem' }}>⚠️</span>
                <div>
                  <span style={{ fontSize: '1rem', fontWeight: '800', color: '#991b1b' }}>
                    Non-Soil Image Detected
                  </span>
                  <div style={{ fontSize: '0.75rem', color: '#b91c1c' }}>Domain Verification: Out-of-Distribution</div>
                </div>
              </div>
              <p style={{ fontSize: '0.85rem', color: '#7f1d1d', margin: '0.5rem 0 0 0', lineHeight: 1.5 }}>
                {cnnResult.rejection_reason || 'The uploaded photo does not match natural soil or agricultural land characteristics. Please upload a clear photo of your field soil.'}
              </p>
            </div>
          ) : (
            <div style={{ marginTop: '1.25rem', padding: '1.25rem', background: '#f0fdf4', borderRadius: '12px', border: '1.5px solid #86efac' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                <div>
                  <span style={{ fontSize: '1.1rem', fontWeight: '800', color: '#14532d' }}>
                    Detected: {cnnResult.detected_soil_type}
                  </span>
                  <div style={{ fontSize: '0.75rem', color: '#166534' }}>Verified against 7 Agro-Soil Classes</div>
                </div>
                <span className="badge badge-green" style={{ fontSize: '0.85rem', padding: '4px 12px' }}>
                  {(cnnResult.confidence * 100).toFixed(1)}% Match
                </span>
              </div>

              {/* Dynamic Visual Pixel Properties */}
              {cnnResult.visual_features && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.5rem', margin: '0.75rem 0', background: 'white', padding: '0.6rem', borderRadius: '8px', border: '1px solid #bbf7d0', fontSize: '0.75rem', color: '#334155' }}>
                  <div>🎨 <b>Hue:</b> {cnnResult.visual_features.mean_hue}</div>
                  <div>💡 <b>Brightness:</b> {cnnResult.visual_features.mean_brightness}</div>
                  <div>🌊 <b>Moisture:</b> {cnnResult.visual_features.estimated_visual_moisture}</div>
                </div>
              )}

              {/* Dynamic Class Probability Bars */}
              <div style={{ marginTop: '0.75rem' }}>
                <div style={{ fontSize: '0.75rem', fontWeight: '700', color: '#166534', marginBottom: '0.35rem' }}>
                  Probability Breakdown across 7 Soil Classes:
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                  {Object.entries(cnnResult.all_probabilities || {}).slice(0, 4).map(([cls, p]) => (
                    <div key={cls} style={{ fontSize: '0.75rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1px', color: '#334155' }}>
                        <span>{cls}</span>
                        <b>{(p * 100).toFixed(1)}%</b>
                      </div>
                      <div style={{ width: '100%', height: '6px', background: '#e2e8f0', borderRadius: '3px', overflow: 'hidden' }}>
                        <div style={{ width: `${p * 100}%`, height: '100%', background: '#16a34a', borderRadius: '3px' }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )
        )}
      </div>

      {/* 2. Manual Soil Chemistry Properties */}
      <div className="card">
        <h3 className="card-title">🧪 2. Manual Soil Lab Test Properties</h3>
        <p className="card-subtitle">
          Input your soil report values. Combined with live local weather, the model recommends optimal crops.
        </p>

        <div className="form-grid-3">
          <div className="form-group">
            <label className="form-label">Nitrogen (N kg/ha)</label>
            <input
              type="number"
              name="nitrogen"
              value={soilData.nitrogen}
              onChange={handleManualChange}
              className="form-input"
              placeholder="e.g. 90"
            />
          </div>

          <div className="form-group">
            <label className="form-label">Phosphorus (P kg/ha)</label>
            <input
              type="number"
              name="phosphorus"
              value={soilData.phosphorus}
              onChange={handleManualChange}
              className="form-input"
              placeholder="e.g. 42"
            />
          </div>

          <div className="form-group">
            <label className="form-label">Potassium (K kg/ha)</label>
            <input
              type="number"
              name="potassium"
              value={soilData.potassium}
              onChange={handleManualChange}
              className="form-input"
              placeholder="e.g. 43"
            />
          </div>
        </div>

        <div className="form-grid-3">
          <div className="form-group">
            <label className="form-label">Soil pH (0 - 14)</label>
            <input
              type="number"
              step="0.1"
              name="ph"
              value={soilData.ph}
              onChange={handleManualChange}
              className="form-input"
              placeholder="e.g. 6.5"
            />
          </div>

          <div className="form-group">
            <label className="form-label">Soil Type</label>
            <select
              name="soil_type"
              value={soilData.soil_type}
              onChange={handleManualChange}
              className="form-select"
            >
              <option value="Alluvial">Alluvial Soil</option>
              <option value="Black">Black Soil (Regur)</option>
              <option value="Red">Red Soil</option>
              <option value="Laterite">Laterite Soil</option>
              <option value="Arid">Arid / Sandy Soil</option>
              <option value="Mountain">Mountain Soil</option>
              <option value="Yellow">Yellow Soil</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Soil Moisture (%)</label>
            <input
              type="number"
              name="moisture"
              value={soilData.moisture}
              onChange={handleManualChange}
              className="form-input"
              placeholder="e.g. 45"
            />
          </div>
        </div>

        <div className="form-grid-3">
          <div className="form-group">
            <label className="form-label">Zinc (Zn ppm)</label>
            <input
              type="number"
              step="0.1"
              name="zinc"
              value={soilData.zinc || ''}
              onChange={handleManualChange}
              className="form-input"
              placeholder="Optional"
            />
          </div>

          <div className="form-group">
            <label className="form-label">Sulphur (S ppm)</label>
            <input
              type="number"
              step="0.1"
              name="sulphur"
              value={soilData.sulphur || ''}
              onChange={handleManualChange}
              className="form-input"
              placeholder="Optional"
            />
          </div>

          <div className="form-group">
            <label className="form-label">EC (dS/m)</label>
            <input
              type="number"
              step="0.01"
              name="electrical_conductivity"
              value={soilData.electrical_conductivity || ''}
              onChange={handleManualChange}
              className="form-input"
              placeholder="Optional"
            />
          </div>
        </div>

        <button
          onClick={onGetRecommendations}
          className="btn-primary"
          disabled={loading}
          style={{ marginTop: '0.75rem', height: '46px' }}
        >
          {loading ? '🧠 Computing ML Ensemble & Agronomic Fit...' : '✨ Recommend Best Crops & Fertilizers'}
        </button>
      </div>
    </div>
  );
};

export default SoilAnalysis;
