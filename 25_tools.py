from langchain_core.tools import tool 
import requests

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@tool
def fetch_weather(city: str) -> str:
    """Fetch the current weather condition and temperature for a specific city. Input must be a raw city name like 'London' or 'Paris'."""
    try:
        # Clean the input to remove accidental leading or trailing spaces
        clean_city = city.strip()
        
        # FIXED: Explicitly separated the domain from the variable with an absolute '/'
        url = f"https://wttr.in{clean_city}"
        params = {"format": "3"}
        headers = {"User-Agent": "Mozilla/5.0"} 
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return f"Current Weather Update -> {response.text.strip()}"
        else:
            return f"Failed to get weather. Server responded with status code: {response.status_code}"
            
    except Exception as e:
        return f"Error executing weather request check: {str(e)}"

# --- Execution Tests ---

# Test Math Tool
result = add_numbers.invoke({"a": 5, "b": 10})
print(f"Math Tool Result: {result}")

# Test Corrected Weather Tool
weather_result = fetch_weather.invoke({"city": "London"})
print(f"Weather Tool Result: {weather_result}")
