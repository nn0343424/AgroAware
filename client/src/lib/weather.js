export async function getWeatherFromCoords(lat, lon, apiKey) {
  const url = `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${apiKey}&units=metric`;

  const res = await fetch(url);
  if (!res.ok) throw new Error("Weather API failed");

  const data = await res.json();
  return {
    temperature: data.main.temp,         
    humidity: data.main.humidity,        
    rainfall: data.rain ? data.rain["1h"] || 0 : 0 
  };
}

export async function reverseGeocode(lat, lon) {
  const url = `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lon}`;
  const res = await fetch(url);
  const data = await res.json();
  return data.address;
}
