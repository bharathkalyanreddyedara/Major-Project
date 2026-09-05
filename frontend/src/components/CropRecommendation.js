import React from 'react';

const CropRecommendation = ({ recommendations, onSelectCrop, selectedCrop }) => {
  if (!recommendations || recommendations.length === 0) return null;

  return (
    <div className="card">
      <h3 className="card-title">🌱 AI Crop & Fertilizer Recommendations</h3>
      <p className="card-subtitle">Ranked by hybrid machine learning suitability (Soil Chemistry + Vision + Weather).</p>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.25rem' }}>
        {recommendations.map((crop, idx) => (
          <div
            key={crop.crop_name}
            style={{
              background: selectedCrop === crop.crop_name ? '#f0fdf4' : '#ffffff',
              border: `2px solid ${selectedCrop === crop.crop_name ? '#16a34a' : '#e2e8f0'}`,
              borderRadius: '12px',
              padding: '1.25rem',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
              boxShadow: '0 2px 8px rgba(0,0,0,0.03)'
            }}
          >
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                <span style={{ fontSize: '1.2rem', fontWeight: '800', color: '#0f172a' }}>
                  #{idx + 1} {crop.crop_name}
                </span>
                <span className="badge badge-green">
                  {crop.suitability_score}% Match
                </span>
              </div>

              <div style={{ fontSize: '0.85rem', color: '#475569', display: 'flex', flexDirection: 'column', gap: '0.35rem', marginBottom: '1rem' }}>
                <div>🗓️ <b>Season:</b> {crop.recommended_season}</div>
                <div>💧 <b>Water:</b> {crop.water_requirement}</div>
                <div>⏳ <b>Duration:</b> {crop.growth_duration_days} Days</div>
                <div>🧪 <b>Fertilizers:</b> {crop.optimal_fertilizers.slice(0, 2).join(', ')}</div>
              </div>
            </div>

            <button
              onClick={() => onSelectCrop(crop.crop_name)}
              className="btn-primary"
              style={{
                background: selectedCrop === crop.crop_name ? '#166534' : 'var(--primary)',
                fontSize: '0.88rem',
                padding: '0.6rem 1rem'
              }}
            >
              {selectedCrop === crop.crop_name ? '✓ Selected for Lifecycle Plan' : '📅 Select & Generate Timeline'}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default CropRecommendation;
