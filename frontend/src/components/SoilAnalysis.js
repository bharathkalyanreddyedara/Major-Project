import React, { useState } from 'react';

const SoilAnalysis = ({
  soilData,
  setSoilData,
  cnnResult,
  setCnnResult,
  onAnalyzeImage,
  onGetRecommendations,
  loading
}) => {
  const [preview, setPreview] = useState(null);
  const [imageFile, setImageFile] = useState(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
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
      {/* 1. Multimodal Soil Vision (CNN) */}
      <div className="card">
        <h3 className="card-title">📷 1. Soil Image Verification (CNN)</h3>
        <p className="card-subtitle">Upload a photo of your field soil for visual classification and validation.</p>

        <label className="dropzone">
          <input
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            style={{ display: 'none' }}
          />
          <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>🌱</div>
          <div style={{ fontWeight: '600', color: '#334155' }}>Click to upload soil photo</div>
          <div style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Supports JPG, PNG, WEBP</div>
        </label>

        {preview && (
          <div style={{ textAlign: 'center' }}>
            <img src={preview} alt="Soil Preview" className="preview-image" />
          </div>
        )}

        {cnnResult && (
          <div style={{ marginTop: '1.25rem', padding: '1rem', background: '#f0fdf4', borderRadius: '10px', border: '1px solid #bbf7d0' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <span style={{ fontWeight: '700', color: '#166534' }}>Detected: {cnnResult.detected_soil_type}</span>
              <span className="badge badge-green">{(cnnResult.confidence * 100).toFixed(1)}% Match</span>
            </div>
            <div style={{ fontSize: '0.82rem', color: '#15803d' }}>
              Texture Roughness: {cnnResult.visual_features?.texture_roughness} | Visual Moisture: {cnnResult.visual_features?.estimated_visual_moisture}
            </div>
          </div>
        )}
      </div>

      {/* 2. Manual Lab Soil Properties */}
      <div className="card">
        <h3 className="card-title">🧪 2. Manual Soil Properties (Lab Data)</h3>
        <p className="card-subtitle">Specify your exact soil chemical and physical test metrics for highest recommendation precision.</p>

        <div className="form-grid-3">
          <div className="form-group">
            <label className="form-label">Nitrogen (N)</label>
            <input
              type="number"
              name="nitrogen"
              value={soilData.nitrogen}
              onChange={handleManualChange}
              className="form-input"
              placeholder="e.g. 90 kg/ha"
            />
          </div>

          <div className="form-group">
            <label className="form-label">Phosphorus (P)</label>
            <input
              type="number"
              name="phosphorus"
              value={soilData.phosphorus}
              onChange={handleManualChange}
              className="form-input"
              placeholder="e.g. 42 kg/ha"
            />
          </div>

          <div className="form-group">
            <label className="form-label">Potassium (K)</label>
            <input
              type="number"
              name="potassium"
              value={soilData.potassium}
              onChange={handleManualChange}
              className="form-input"
              placeholder="e.g. 43 kg/ha"
            />
          </div>
        </div>

        <div className="form-grid-3">
          <div className="form-group">
            <label className="form-label">Soil pH (0-14)</label>
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
            <label className="form-label">Moisture (%)</label>
            <input
              type="number"
              name="moisture"
              value={soilData.moisture}
              onChange={handleManualChange}
              className="form-input"
              placeholder="e.g. 40%"
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
          style={{ marginTop: '0.75rem' }}
        >
          {loading ? 'Analyzing & Recommending...' : '✨ Recommend Best Crops & Fertilizers'}
        </button>
      </div>
    </div>
  );
};

export default SoilAnalysis;
