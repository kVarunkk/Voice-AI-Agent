import httpx

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODE_MAP = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow",
    80: "rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    95: "thunderstorm", 96: "thunderstorm with hail",
}

async def get_weather(location: str) -> dict:
    async with httpx.AsyncClient(timeout=5.0) as client:
        geo_resp = await client.get(GEOCODE_URL, params={"name": location, "count": 1})
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()

        results = geo_data.get("results")
        if not results:
            return {"error": f"location not found: {location}"}

        place = results[0]
        lat, lon = place["latitude"], place["longitude"]
        resolved_name = place.get("name", location)
        country = place.get("country", "")

        weather_resp = await client.get(WEATHER_URL, params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,weather_code,wind_speed_10m",
            "temperature_unit": "fahrenheit",
        })
        weather_resp.raise_for_status()
        weather_data = weather_resp.json()

        current = weather_data.get("current", {})
        code = current.get("weather_code")

        return {
            "location": f"{resolved_name}, {country}".strip(", "),
            "temp_f": current.get("temperature_2m"),
            "wind_speed_mph": current.get("wind_speed_10m"),
            "condition": WEATHER_CODE_MAP.get(code, "unknown"),
        }

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather conditions for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name, e.g. 'Austin' or 'Paris'"}
                },
                "required": ["location"],
            },
        },
    },
]

TOOL_IMPLS = {"get_weather": get_weather}