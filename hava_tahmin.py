import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Hava durumu verisini oku
file_path = r'C:\Users\berra\Downloads\archive\seattle-weather.csv'
df = pd.read_csv(file_path)

# Veri ön işleme: 'temperature' ve 'humidity' sütunlarını numerik hale getir
df['temperature'] = pd.to_numeric(df['temp_max'], errors='coerce')
df['humidity'] = pd.to_numeric(df['precipitation'], errors='coerce')

# Hava durumu değişkenleri (inputlar)
temperature = ctrl.Antecedent(np.arange(-10, 50, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(0, 100, 1), 'humidity')
wind_speed = ctrl.Antecedent(np.arange(0, 30, 1), 'wind_speed')

# Hava durumu kategorileri (output)
weather = ctrl.Consequent(np.arange(0, 101, 1), 'weather')

# Üyelik fonksiyonlarını tanımla
temperature['low'] = fuzz.trapmf(temperature.universe, [-10, -10, 0, 10])
temperature['medium'] = fuzz.trimf(temperature.universe, [0, 20, 40])
temperature['high'] = fuzz.trapmf(temperature.universe, [20, 40, 50, 50])

humidity['low'] = fuzz.trapmf(humidity.universe, [0, 0, 10, 30])
humidity['medium'] = fuzz.trimf(humidity.universe, [10, 50, 80])
humidity['high'] = fuzz.trapmf(humidity.universe, [50, 80, 100, 100])

wind_speed['low'] = fuzz.trapmf(wind_speed.universe, [0, 0, 5, 10])
wind_speed['medium'] = fuzz.trimf(wind_speed.universe, [5, 15, 25])
wind_speed['high'] = fuzz.trapmf(wind_speed.universe, [15, 25, 30, 30])

weather['sunny'] = fuzz.trapmf(weather.universe, [0, 0, 30, 50])
weather['cloudy'] = fuzz.trimf(weather.universe, [20, 50, 80])
weather['rainy'] = fuzz.trimf(weather.universe, [40, 60, 100])
weather['stormy'] = fuzz.trapmf(weather.universe, [60, 100, 100, 100])

# Kurallar
rule1 = ctrl.Rule(temperature['high'] & humidity['low'], weather['sunny'])
rule2 = ctrl.Rule(temperature['low'] & wind_speed['high'], weather['stormy'])
rule3 = ctrl.Rule(humidity['high'] & wind_speed['low'], weather['rainy'])

# Bulanık mantık sistemi tanımla ve kuralları ekle
weather_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
weather_simulation = ctrl.ControlSystemSimulation(weather_ctrl)

# Simülasyonu çalıştır
# Örneğin, belli bir hava durumu için tahmin yapmak
weather_simulation.input['temperature'] = 22
weather_simulation.input['humidity'] = 60
weather_simulation.input['wind_speed'] = 5

weather_simulation.compute()
print("Tahmin edilen hava durumu:", weather_simulation.output['weather'])

# Tahmin edilen hava durumu kategorisi
if weather_simulation.output['weather'] <= 30:
    print("Tahmin edilen hava durumu: Sunny")
elif weather_simulation.output['weather'] <= 50:
    print("Tahmin edilen hava durumu: Cloudy")
elif weather_simulation.output['weather'] <= 80:
    print("Tahmin edilen hava durumu: Rainy")
else:
    print("Tahmin edilen hava durumu: Stormy")
