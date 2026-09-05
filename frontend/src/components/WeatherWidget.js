import React from 'react';

const WeatherWidget = ({ weather, city, setCity, onRefresh }) => {
  if (!weather) return null;

  return (
    <div className="card" style={{ background: 'linear-gradient(135deg, #065f46 0%, #047857 100%)', color: 'white', border: 'none' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
            <span style={{ fontSize: '1.25rem', fontWeight: '700' }}>📍 {weather.city}</span>
            <span className="badge" style={{ background: 'rgba(255,255,255,0.2)', color: 'white' }}>
              {weather.is_live ? 'Live Weather' : 'Seasonal Climatology'}
            </span>
          </div>
          <p style={{ opacity: 0.85, fontSize: '0.9rem' }}>{weather.weather_condition}</p>
        </div>

        <div style={{ display: 'flex', gap: '2rem', alignItems: 'center' }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '1.75rem', fontWeight: '800' }}>{weather.temperature}°C</div>
            <div style={{ fontSize: '0.75rem', opacity: 0.8 }}>Temperature</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '1.75rem', fontWeight: '800' }}>{weather.humidity}%</div>
            <div style={{ fontSize: '0.75rem', opacity: 0.8 }}>Humidity</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '1.75rem', fontWeight: '800' }}>{weather.rainfall} mm</div>
            <div style={{ fontSize: '0.75rem', opacity: 0.8 }}>Precipitation</div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <input
            type="text"
            placeholder="Change location..."
            value={city}
            onChange={(e) => setCity(e.target.value)}
            style={{
              padding: '6px 12px',
              borderRadius: '8px',
              border: 'none',
              background: 'rgba(255,255,255,0.9)',
              color: '#0f172a',
              fontSize: '0.85rem'
            }}
          />
          <button
            onClick={onRefresh}
            style={{
              padding: '6px 12px',
              borderRadius: '8px',
              border: '1px solid rgba(255,255,255,0.4)',
              background: 'rgba(255,255,255,0.2)',
              color: 'white',
              cursor: 'pointer',
              fontWeight: '600',
              fontSize: '0.85rem'
            }}
          >
            Update
          </button>
        </div>
      </div>
    </div>
  );
};

export default WeatherWidget;
