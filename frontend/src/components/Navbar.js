import React from 'react';

const Navbar = ({ activeTab, setActiveTab, unreadCount = 0 }) => {
  const tabs = [
    { id: 'advisor', label: '🌾 Soil & Crop Advisor' },
    { id: 'timeline', label: '📅 Lifecycle Timeline' },
    { id: 'assistant', label: '🤖 AI Farm Assistant' },
    { id: 'alerts', label: `🔔 Alerts ${unreadCount > 0 ? `(${unreadCount})` : ''}` }
  ];

  return (
    <header className="navbar">
      <div className="nav-brand">
        <span>🌱 AgriVision AI</span>
        <span className="brand-badge">Multimodal Farm Intelligence</span>
      </div>

      <nav className="nav-tabs">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>
    </header>
  );
};

export default Navbar;
