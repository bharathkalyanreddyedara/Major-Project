import React, { useState } from 'react';

const CropTimeline = ({ timeline, onGenerateTimeline, selectedCrop, soilType, location }) => {
  const [sowingDate, setSowingDate] = useState(
    timeline ? timeline.sowing_date : new Date().toISOString().split('T')[0]
  );

  const handleDateChange = (newDate) => {
    setSowingDate(newDate);
    onGenerateTimeline(selectedCrop || timeline?.crop_name || 'Rice', newDate);
  };

  if (!timeline) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '3.5rem 1.5rem' }}>
        <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>📅</div>
        <h3 style={{ fontSize: '1.4rem', fontWeight: '800' }}>No Active Cultivation Timeline</h3>
        <p style={{ color: '#64748b', maxWidth: '500px', margin: '0.5rem auto 1.5rem auto' }}>
          Select a recommended crop from the advisor tab to generate an automated, stage-aware cultivation schedule.
        </p>
        <button
          onClick={() => onGenerateTimeline(selectedCrop || 'Rice', sowingDate)}
          className="btn-primary"
          style={{ maxWidth: '320px', margin: '0 auto' }}
        >
          Generate Cultivation Schedule for {selectedCrop || 'Rice'}
        </button>
      </div>
    );
  }

  const progressPct = Math.min(100, Math.round((timeline.current_day / timeline.total_duration_days) * 100));

  return (
    <div className="card">
      {/* Header & Date Controls */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem', borderBottom: '1px solid #e2e8f0', paddingBottom: '1.25rem', marginBottom: '1.5rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <h3 className="card-title" style={{ margin: 0, fontSize: '1.4rem' }}>
              📅 {timeline.crop_name} Cultivation Timeline
            </h3>
            <span className="badge badge-green">Stage: {timeline.current_stage}</span>
          </div>
          <p style={{ color: '#64748b', fontSize: '0.88rem', marginTop: '0.35rem' }}>
            Expected Harvest Date: <b>{timeline.expected_harvest_date}</b> (Total Duration: {timeline.total_duration_days} Days)
          </p>
        </div>

        {/* Dynamic Sowing Date Picker */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', background: '#f8fafc', padding: '0.5rem 1rem', borderRadius: '10px', border: '1px solid #e2e8f0' }}>
          <label style={{ fontSize: '0.85rem', fontWeight: '700', color: '#334155' }}>Sowing Date:</label>
          <input
            type="date"
            value={sowingDate}
            onChange={(e) => handleDateChange(e.target.value)}
            className="form-input"
            style={{ width: 'auto', padding: '4px 10px', fontSize: '0.85rem' }}
          />
        </div>
      </div>

      {/* Dynamic Lifecycle Progress Bar */}
      <div style={{ marginBottom: '2rem', background: '#f0fdf4', padding: '1rem 1.25rem', borderRadius: '12px', border: '1px solid #bbf7d0' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
          <span style={{ fontSize: '0.9rem', fontWeight: '700', color: '#166534' }}>
            Growth Progress: Day {timeline.current_day} of {timeline.total_duration_days} Days
          </span>
          <span style={{ fontSize: '0.9rem', fontWeight: '800', color: '#166534' }}>
            {progressPct}% Completed
          </span>
        </div>
        <div style={{ width: '100%', height: '10px', background: '#dcfce7', borderRadius: '5px', overflow: 'hidden' }}>
          <div style={{ width: `${progressPct}%`, height: '100%', background: 'linear-gradient(90deg, #22c55e, #15803d)', borderRadius: '5px', transition: 'width 0.4s ease' }} />
        </div>
      </div>

      {/* Dynamic Stage-by-Stage Stepper */}
      <div className="timeline-stepper">
        {timeline.stages.map((st) => (
          <div
            key={st.stage_id}
            className={`timeline-item ${st.status === 'current' ? 'active' : ''}`}
            style={{
              borderColor: st.status === 'current' ? '#16a34a' : (st.status === 'completed' ? '#cbd5e1' : '#e2e8f0'),
              boxShadow: st.status === 'current' ? '0 4px 15px rgba(22, 163, 74, 0.1)' : 'none'
            }}
          >
            <div
              className="timeline-marker"
              style={{
                borderColor: st.status === 'completed' ? '#16a34a' : (st.status === 'current' ? '#16a34a' : '#cbd5e1'),
                background: st.status === 'completed' ? '#16a34a' : (st.status === 'current' ? '#ffffff' : '#ffffff')
              }}
            >
              {st.status === 'completed' && <span style={{ color: '#ffffff', fontSize: '12px', fontWeight: '900' }}>✓</span>}
              {st.status === 'current' && <span style={{ color: '#16a34a', fontSize: '10px' }}>●</span>}
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '0.5rem' }}>
              <h4 style={{ fontWeight: '800', fontSize: '1.1rem', color: '#0f172a' }}>
                Stage {st.stage_id}: {st.stage_name}
              </h4>
              <span className={`badge ${st.status === 'completed' ? 'badge-green' : (st.status === 'current' ? 'badge-yellow' : 'badge-blue')}`} style={{ fontSize: '0.8rem', padding: '4px 10px' }}>
                Days {st.start_day} - {st.end_day} ({st.status.toUpperCase()})
              </span>
            </div>

            <p style={{ fontSize: '0.8rem', color: '#64748b', fontStyle: 'italic', marginBottom: '0.75rem' }}>
              {st.critical_notes}
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.25rem', marginTop: '0.75rem', fontSize: '0.88rem' }}>
              <div style={{ background: '#f8fafc', padding: '0.85rem', borderRadius: '8px', border: '1px solid #f1f5f9' }}>
                <b style={{ color: '#166534', display: 'block', marginBottom: '0.35rem' }}>🌾 Key Stage Activities:</b>
                <ul style={{ paddingLeft: '1.2rem', color: '#334155', display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                  {st.activities.map((act, i) => <li key={i}>{act}</li>)}
                </ul>
              </div>

              <div style={{ background: '#f8fafc', padding: '0.85rem', borderRadius: '8px', border: '1px solid #f1f5f9', display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
                <div>
                  <b style={{ color: '#0369a1' }}>💧 Irrigation:</b> <span style={{ color: '#334155' }}>{st.irrigation_schedule}</span>
                </div>
                <div>
                  <b style={{ color: '#854d0e' }}>🧪 Fertilizer:</b> <span style={{ color: '#334155' }}>{st.fertilizer_advice}</span>
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
