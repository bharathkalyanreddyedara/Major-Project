import React, { useState, useEffect } from 'react';
import './App.css';
import Navbar from './components/Navbar';
import WeatherWidget from './components/WeatherWidget';
import SoilAnalysis from './components/SoilAnalysis';
import CropRecommendation from './components/CropRecommendation';
import CropTimeline from './components/CropTimeline';
import NotificationPanel from './components/NotificationPanel';
import AIChatAssistant from './components/AIChatAssistant';

const API_BASE = 'http://localhost:8000/api';

function App() {
  const [activeTab, setActiveTab] = useState('advisor');
  const [city, setCity] = useState('Hyderabad');
  const [weather, setWeather] = useState(null);
  const [loading, setLoading] = useState(false);
  const [imageLoading, setImageLoading] = useState(false);

  // Dynamic Soil and Lab Parameters State
  const [soilData, setSoilData] = useState({
    nitrogen: 90,
    phosphorus: 42,
    potassium: 43,
    ph: 6.5,
    soil_type: 'Black',
    moisture: 45,
    zinc: null,
    sulphur: null,
    electrical_conductivity: null
  });

  const [cnnResult, setCnnResult] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [selectedCrop, setSelectedCrop] = useState('Rice');
  const [timeline, setTimeline] = useState(null);
  const [notifications, setNotifications] = useState([]);

  // Live Chat State
  const [chatMessages, setChatMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! I am your AI Agricultural Knowledge Assistant. Ask me anything about crop cultivation, stage-specific fertilizer doses, pest management, or current weather impacts.',
      sources: []
    }
  ]);
  const [chatLoading, setChatLoading] = useState(false);

  // 1. Fetch Dynamic Weather & Notifications
  const fetchWeatherAndAlerts = async () => {
    try {
      const res = await fetch(`${API_BASE}/weather?city=${encodeURIComponent(city)}`);
      if (res.ok) {
        const data = await res.json();
        setWeather(data);
      }
    } catch (e) {
      console.log('Weather fetch notice:', e);
    }

    try {
      const notifRes = await fetch(`${API_BASE}/notifications?city=${encodeURIComponent(city)}`);
      if (notifRes.ok) {
        const notifs = await notifRes.json();
        setNotifications(notifs);
      }
    } catch (e) {
      console.log('Notifications notice:', e);
    }
  };

  useEffect(() => {
    fetchWeatherAndAlerts();
  }, []);

  // 2. Real-Time Image Feature Extraction & Classification
  const handleAnalyzeImage = async (file) => {
    setImageLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/soil/analyze`, {
        method: 'POST',
        body: formData
      });
      if (res.ok) {
        const data = await res.json();
        setCnnResult(data);
        if (data.detected_soil_type) {
          const cleanName = data.detected_soil_type.replace(' Soil', '').trim();
          setSoilData(prev => ({
            ...prev,
            soil_type: cleanName
          }));
        }
      } else {
        const err = await res.json();
        alert(`Error analyzing image: ${err.detail || 'Server error'}`);
      }
    } catch (e) {
      console.error('Soil analysis error:', e);
      alert('Unable to connect to backend server. Make sure uvicorn is running on port 8000.');
    } finally {
      setImageLoading(false);
    }
  };

  // 3. Dynamic Crop Recommendations
  const handleGetRecommendations = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/crop/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          soil_properties: soilData,
          city: city,
          temperature: weather?.temperature,
          humidity: weather?.humidity,
          rainfall: weather?.rainfall
        })
      });

      if (res.ok) {
        const data = await res.json();
        setRecommendations(data.recommendations || []);
        if (data.recommendations && data.recommendations.length > 0) {
          setSelectedCrop(data.recommendations[0].crop_name);
        }
      } else {
        const err = await res.json();
        alert(`Recommendation error: ${err.detail || 'Server error'}`);
      }
    } catch (e) {
      console.error('Crop recommendation error:', e);
      alert('Could not fetch recommendations. Ensure backend is running.');
    } finally {
      setLoading(false);
    }
  };

  // 4. Dynamic Timeline Generator
  const handleGenerateTimeline = async (cropName, sowingDate = new Date().toISOString().split('T')[0]) => {
    setSelectedCrop(cropName);
    try {
      const res = await fetch(`${API_BASE}/timeline/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          crop_name: cropName,
          sowing_date: sowingDate,
          soil_type: soilData.soil_type || 'Alluvial',
          location: city
        })
      });

      if (res.ok) {
        const data = await res.json();
        setTimeline(data);
        setActiveTab('timeline');
      }
    } catch (e) {
      console.error('Timeline generation error:', e);
    }
  };

  // 5. Live RAG Chat Query to Agricultural Assistant
  const handleSendMessage = async (queryText) => {
    const userMsg = { role: 'user', content: queryText };
    setChatMessages(prev => [...prev, userMsg]);
    setChatLoading(true);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: queryText,
          crop_context: selectedCrop,
          soil_context: `${soilData.soil_type} Soil (pH: ${soilData.ph}, N: ${soilData.nitrogen}, P: ${soilData.phosphorus}, K: ${soilData.potassium})`,
          growth_stage_context: timeline?.current_stage || 'Active Field Growth',
          weather_context: `${weather?.temperature}°C, ${weather?.weather_condition}`,
          history: chatMessages
        })
      });

      if (res.ok) {
        const data = await res.json();
        setChatMessages(prev => [
          ...prev,
          {
            role: 'assistant',
            content: data.answer,
            sources: data.grounded_sources
          }
        ]);
      }
    } catch (e) {
      setChatMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: 'Unable to connect to AI server. Please make sure the backend is active at port 8000.',
          sources: []
        }
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        unreadCount={notifications.length}
      />

      <main className="main-content">
        <WeatherWidget
          weather={weather}
          city={city}
          setCity={setCity}
          onRefresh={fetchWeatherAndAlerts}
        />

        {activeTab === 'advisor' && (
          <>
            <SoilAnalysis
              soilData={soilData}
              setSoilData={setSoilData}
              cnnResult={cnnResult}
              setCnnResult={setCnnResult}
              onAnalyzeImage={handleAnalyzeImage}
              onGetRecommendations={handleGetRecommendations}
              loading={loading}
              imageLoading={imageLoading}
            />
            <CropRecommendation
              recommendations={recommendations}
              onSelectCrop={handleGenerateTimeline}
              selectedCrop={selectedCrop}
            />
          </>
        )}

        {activeTab === 'timeline' && (
          <CropTimeline
            timeline={timeline}
            onGenerateTimeline={handleGenerateTimeline}
            selectedCrop={selectedCrop}
            soilType={soilData.soil_type}
            location={city}
          />
        )}

        {activeTab === 'assistant' && (
          <AIChatAssistant
            onSendMessage={handleSendMessage}
            messages={chatMessages}
            loading={chatLoading}
            activeCrop={selectedCrop}
            currentStage={timeline?.current_stage}
            weather={weather}
          />
        )}

        {activeTab === 'alerts' && (
          <NotificationPanel notifications={notifications} />
        )}
      </main>
    </div>
  );
}

export default App;
