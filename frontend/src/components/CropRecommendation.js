import React from 'react';

const CropRecommendation = ({ recommendations, onSelectCrop, selectedCrop }) => {
  if (!recommendations || recommendations.length === 0) return null;

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div>
          <h3 className="card-title">🌱 AI Crop & Fertilizer Recommendations</h3>
          <p className="card-subtitle" style={{ marginBottom: 0 }}>
            Ranked by trained Ensemble ML Model + Agronomic Soil & Weather compatibility.
          </p>
        </div>
        <span className="badge badge-green" style={{ fontSize: '0.8rem', padding: '6px 12px' }}>
          {recommendations.length} Best Matched Crops Found
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '1.25rem' }}>
        {recommendations.map((crop, idx) => (
          <div
            key={crop.crop_name}
            style={{
              background: selectedCrop === crop.crop_name ? '#f0fdf4' : '#ffffff',
              border: `2px solid ${selectedCrop === crop.crop_name ? '#16a34a' : '#e2e8f0'}`,
              borderRadius: '14px',
              padding: '1.35rem',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
              boxShadow: selectedCrop === crop.crop_name ? '0 8px 20px rgba(22, 163, 74, 0.12)' : '0 2px 8px rgba(0,0,0,0.04)',
              transition: 'all 0.2s ease'
            }}
          >
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem' }}>
                <div>
                  <span style={{ fontSize: '1.25rem', fontWeight: '800', color: '#0f172a' }}>
                    #{idx + 1} {crop.crop_name}
                  </span>
                  <div style={{ fontSize: '0.75rem', color: '#64748b' }}>
                    Model Confidence: <b>{(crop.confidence * 100).toFixed(1)}%</b>
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <span className={`badge ${crop.suitability_score > 70 ? 'badge-green' : 'badge-yellow'}`} style={{ fontSize: '0.85rem', padding: '4px 10px' }}>
                    {crop.suitability_score}% Overall Fit
                  </span>
                </div>
              </div>

              <div style={{ fontSize: '0.86rem', color: '#334155', display: 'flex', flexDirection: 'column', gap: '0.45rem', margin: '1rem 0', background: '#f8fafc', padding: '0.85rem', borderRadius: '10px', border: '1px solid #f1f5f9' }}>
                <div>🗓️ <b>Season:</b> {crop.recommended_season}</div>
                <div>💧 <b>Water Need:</b> {crop.water_requirement}</div>
                <div>⏳ <b>Growth Duration:</b> {crop.growth_duration_days} Days to Harvest</div>
                <div>🪴 <b>Soil Compatibility:</b> <span style={{ color: crop.soil_compatibility === 'Excellent' ? '#16a34a' : '#ca8a04', fontWeight: '700' }}>{crop.soil_compatibility}</span></div>
                <div>🧪 <b>Fertilizer Advice:</b> <span style={{ color: '#15803d', fontWeight: '600' }}>{crop.optimal_fertilizers.slice(0, 3).join(', ')}</span></div>
              </div>
            </div>

            <button
              onClick={() => onSelectCrop(crop.crop_name)}
              className="btn-primary"
              style={{
                background: selectedCrop === crop.crop_name ? '#166534' : 'var(--primary)',
                fontSize: '0.9rem',
                padding: '0.75rem 1rem',
                marginTop: '0.5rem'
              }}
            >
              {selectedCrop === crop.crop_name ? '✓ Selected (View Timeline)' : '📅 Select & Generate Cultivation Schedule'}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default CropRecommendation;
