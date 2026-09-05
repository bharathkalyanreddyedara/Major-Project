import React from 'react';

const NotificationPanel = ({ notifications }) => {
  return (
    <div className="card">
      <h3 className="card-title">🔔 Proactive Farm Alerts & Timely Reminders</h3>
      <p className="card-subtitle">Automated stage-aware reminders, weather alarms, and irrigation/fertilizer triggers.</p>

      {(!notifications || notifications.length === 0) ? (
        <p style={{ color: '#64748b' }}>No new notifications at this time.</p>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {notifications.map((notif) => (
            <div
              key={notif.id}
              style={{
                padding: '1rem 1.25rem',
                borderRadius: '10px',
                border: '1px solid',
                borderColor: notif.severity === 'warning' ? '#fde047' : '#bbf7d0',
                background: notif.severity === 'warning' ? '#fefce8' : '#f0fdf4',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'flex-start',
                gap: '1rem'
              }}
            >
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
                  <span style={{ fontWeight: '700', color: notif.severity === 'warning' ? '#854d0e' : '#166534' }}>
                    {notif.title}
                  </span>
                  <span className={`badge ${notif.severity === 'warning' ? 'badge-yellow' : 'badge-green'}`}>
                    {notif.category || 'General'}
                  </span>
                </div>
                <p style={{ fontSize: '0.9rem', color: '#334155', margin: 0 }}>{notif.message}</p>
              </div>

              <span style={{ fontSize: '0.75rem', color: '#94a3b8', whiteSpace: 'nowrap' }}>
                {notif.timestamp}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default NotificationPanel;
