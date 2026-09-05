import React, { useState } from 'react';

const CropTimeline = ({ timeline, onGenerateTimeline, selectedCrop }) => {
  const [sowingDate, setSowingDate] = useState(new Date().toISOString().split('T')[0]);

  if (!timeline) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '3rem 1.5rem' }}>
        <div style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>📅</div>
        <h3>No Active Crop Timeline</h3>
        <p style={{ color: '#64748b', marginBottom: '1.5rem' }}>
          Select a recommended crop from the advisor tab to generate an automated cultivation schedule.
        </p>
        <button
          onClick={() => onGenerateTimeline(selectedCrop || 'Rice', sowingDate)}
          className="btn-primary"
          style={{ maxWidth: '300px', margin: '0 auto' }}
        >
          Generate Default Timeline for {selectedCrop || 'Rice'}
        </button>
      </div>
    );
  }

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem', marginBottom: '1.5rem' }}>
        <div>
          <h3 className="card-title">📅 Cultivation Timeline: {timeline.crop_name}</h3>
          <p className="card-subtitle" style={{ marginBottom: 0 }}>
            Sown on <b>{timeline.sowing_date}</b> | Expected Harvest: <b>{timeline.expected_harvest_date}</b> ({timeline.total_duration_days} Days)
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <span className="badge badge-blue">Day {timeline.current_day} of {timeline.total_duration_days}</span>
          <span className="badge badge-green">Stage: {timeline.current_stage}</span>
        </div>
      </div>

      {/* Timeline Stages */}
      <div className="timeline-stepper">
        {timeline.stages.map((st) => (
          <div
            key={st.stage_id}
            className={`timeline-item ${st.status === 'current' ? 'active' : ''}`}
          >
            <div className="timeline-marker">
              {st.status === 'completed' && <span style={{ color: '#16a34a', fontSize: '10px' }}>✓</span>}
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <h4 style={{ fontWeight: '700', fontSize: '1.05rem', color: '#0f172a' }}>
                Stage {st.stage_id}: {st.stage_name}
              </h4>
              <span className={`badge ${st.status === 'completed' ? 'badge-green' : (st.status === 'current' ? 'badge-yellow' : 'badge-blue')}`}>
                Days {st.start_day} - {st.end_day} ({st.status.toUpperCase()})
              </span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '0.75rem', fontSize: '0.88rem' }}>
              <div>
                <b style={{ color: '#166534' }}>🌾 Key Activities:</b>
                <ul style={{ paddingLeft: '1.2rem', marginTop: '0.25rem', color: '#334155' }}>
                  {st.activities.map((act, i) => <li key={i}>{act}</li>)}
                </ul>
              </div>

              <div>
                <div style={{ marginBottom: '0.35rem' }}>
                  <b style={{ color: '#0369a1' }}>💧 Irrigation:</b> <span style={{ color: '#334155' }}>{st.irrigation_schedule}</span>
                </div>
                <div style={{ marginBottom: '0.35rem' }}>
                  <b style={{ color: '#854d0e' }}>🧪 Fertilizer Top-Dressing:</b> <span style={{ color: '#334155' }}>{st.fertilizer_advice}</span>
                </div>
                <div>
                  <b style={{ color: '#b91c1c' }}>🐛 Pest/Disease Watch:</b> <span style={{ color: '#334155' }}>{st.pest_disease_watch}</span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default CropTimeline;
