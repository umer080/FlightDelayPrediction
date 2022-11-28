import json
import joblib
import requests
import numpy as np
from pickle import load
from datetime import datetime
from flask import Flask, jsonify, request
# from sklearn.preprocessing import MinMaxScaler, StandardScaler

app = Flask(__name__)

model = load(open('model.pkl', 'rb'))
scaler = load(open('scaler.pkl', 'rb'))
minmax = load(open('minmax.pkl', 'rb'))

@app.route("/predict", methods=["POST"])
def index():
    data = request.json
    print(data)
    departure_time=data['time']
    original_origin=data['origin']
    destination=data['dest']
    airline=data['airline']

    carrier_str = '{"DL":{"ARR_DEL15":51331.0,"CARRIER_cat":15},"EV":{"ARR_DEL15":15236.0,"CARRIER_cat":14},"WN":{"ARR_DEL15":14631.0,"CARRIER_cat":13},"AA":{"ARR_DEL15":4533.0,"CARRIER_cat":12},"OO":{"ARR_DEL15":3930.0,"CARRIER_cat":11},"NK":{"ARR_DEL15":3187.0,"CARRIER_cat":10},"UA":{"ARR_DEL15":1791.0,"CARRIER_cat":9},"F9":{"ARR_DEL15":1490.0,"CARRIER_cat":8},"B6":{"ARR_DEL15":295.0,"CARRIER_cat":7},"AS":{"ARR_DEL15":165.0,"CARRIER_cat":6},"YX":{"ARR_DEL15":9.0,"CARRIER_cat":5},"OH":{"ARR_DEL15":4.0,"CARRIER_cat":4},"YV":{"ARR_DEL15":2.0,"CARRIER_cat":2},"MQ":{"ARR_DEL15":2.0,"CARRIER_cat":2},"9E":{"ARR_DEL15":1.0,"CARRIER_cat":1},"VX":{"ARR_DEL15":0.0,"CARRIER_cat":0},"G4":{"ARR_DEL15":0.0,"CARRIER_cat":0}}'
    carrier_json = json.loads(carrier_str)
    airline = carrier_json[airline]['CARRIER_cat']

    origin_str = '{"LGA":{"ARR_DEL15":3060.0,"ORIGIN_cat":24},"ORD":{"ARR_DEL15":2674.0,"ORIGIN_cat":24},"LAX":{"ARR_DEL15":2593.0,"ORIGIN_cat":24},"MCO":{"ARR_DEL15":2303.0,"ORIGIN_cat":24},"FLL":{"ARR_DEL15":2302.0,"ORIGIN_cat":24},"PHL":{"ARR_DEL15":2106.0,"ORIGIN_cat":24},"DFW":{"ARR_DEL15":2085.0,"ORIGIN_cat":24},"TPA":{"ARR_DEL15":1930.0,"ORIGIN_cat":23},"EWR":{"ARR_DEL15":1925.0,"ORIGIN_cat":23},"BWI":{"ARR_DEL15":1889.0,"ORIGIN_cat":23},"DEN":{"ARR_DEL15":1827.0,"ORIGIN_cat":23},"BOS":{"ARR_DEL15":1827.0,"ORIGIN_cat":23},"CLT":{"ARR_DEL15":1614.0,"ORIGIN_cat":23},"DTW":{"ARR_DEL15":1608.0,"ORIGIN_cat":23},"MIA":{"ARR_DEL15":1598.0,"ORIGIN_cat":22},"DCA":{"ARR_DEL15":1505.0,"ORIGIN_cat":22},"MSY":{"ARR_DEL15":1482.0,"ORIGIN_cat":22},"RDU":{"ARR_DEL15":1462.0,"ORIGIN_cat":22},"DAL":{"ARR_DEL15":1301.0,"ORIGIN_cat":22},"MDW":{"ARR_DEL15":1289.0,"ORIGIN_cat":22},"HOU":{"ARR_DEL15":1271.0,"ORIGIN_cat":21},"IAH":{"ARR_DEL15":1260.0,"ORIGIN_cat":21},"LAS":{"ARR_DEL15":1165.0,"ORIGIN_cat":21},"JAX":{"ARR_DEL15":1156.0,"ORIGIN_cat":21},"MSP":{"ARR_DEL15":1130.0,"ORIGIN_cat":21},"STL":{"ARR_DEL15":1059.0,"ORIGIN_cat":21},"SFO":{"ARR_DEL15":1038.0,"ORIGIN_cat":21},"IND":{"ARR_DEL15":984.0,"ORIGIN_cat":20},"RIC":{"ARR_DEL15":961.0,"ORIGIN_cat":20},"PBI":{"ARR_DEL15":947.0,"ORIGIN_cat":20},"BTR":{"ARR_DEL15":913.0,"ORIGIN_cat":20},"GSP":{"ARR_DEL15":839.0,"ORIGIN_cat":20},"AUS":{"ARR_DEL15":838.0,"ORIGIN_cat":20},"RSW":{"ARR_DEL15":834.0,"ORIGIN_cat":19},"JFK":{"ARR_DEL15":834.0,"ORIGIN_cat":19},"MCI":{"ARR_DEL15":833.0,"ORIGIN_cat":19},"PHX":{"ARR_DEL15":831.0,"ORIGIN_cat":19},"IAD":{"ARR_DEL15":812.0,"ORIGIN_cat":19},"BNA":{"ARR_DEL15":801.0,"ORIGIN_cat":19},"CMH":{"ARR_DEL15":800.0,"ORIGIN_cat":19},"CHA":{"ARR_DEL15":797.0,"ORIGIN_cat":18},"HPN":{"ARR_DEL15":751.0,"ORIGIN_cat":18},"CLE":{"ARR_DEL15":745.0,"ORIGIN_cat":18},"AGS":{"ARR_DEL15":736.0,"ORIGIN_cat":18},"SAT":{"ARR_DEL15":735.0,"ORIGIN_cat":18},"MEM":{"ARR_DEL15":725.0,"ORIGIN_cat":18},"MOB":{"ARR_DEL15":719.0,"ORIGIN_cat":18},"BHM":{"ARR_DEL15":716.0,"ORIGIN_cat":17},"MKE":{"ARR_DEL15":703.0,"ORIGIN_cat":17},"AVL":{"ARR_DEL15":693.0,"ORIGIN_cat":17},"SAV":{"ARR_DEL15":691.0,"ORIGIN_cat":17},"LEX":{"ARR_DEL15":675.0,"ORIGIN_cat":17},"MGM":{"ARR_DEL15":658.0,"ORIGIN_cat":17},"CAE":{"ARR_DEL15":657.0,"ORIGIN_cat":16},"PIT":{"ARR_DEL15":656.0,"ORIGIN_cat":16},"SEA":{"ARR_DEL15":648.0,"ORIGIN_cat":16},"SHV":{"ARR_DEL15":620.0,"ORIGIN_cat":16},"ORF":{"ARR_DEL15":596.0,"ORIGIN_cat":16},"GNV":{"ARR_DEL15":590.0,"ORIGIN_cat":16},"SLC":{"ARR_DEL15":578.0,"ORIGIN_cat":16},"CHS":{"ARR_DEL15":575.0,"ORIGIN_cat":15},"SDF":{"ARR_DEL15":524.0,"ORIGIN_cat":15},"GSO":{"ARR_DEL15":521.0,"ORIGIN_cat":15},"TRI":{"ARR_DEL15":520.0,"ORIGIN_cat":15},"VPS":{"ARR_DEL15":516.0,"ORIGIN_cat":15},"HSV":{"ARR_DEL15":507.0,"ORIGIN_cat":14},"JAN":{"ARR_DEL15":507.0,"ORIGIN_cat":14},"TYS":{"ARR_DEL15":504.0,"ORIGIN_cat":14},"TLH":{"ARR_DEL15":494.0,"ORIGIN_cat":14},"CRW":{"ARR_DEL15":494.0,"ORIGIN_cat":14},"CVG":{"ARR_DEL15":482.0,"ORIGIN_cat":14},"ROA":{"ARR_DEL15":466.0,"ORIGIN_cat":14},"BDL":{"ARR_DEL15":465.0,"ORIGIN_cat":14},"SBN":{"ARR_DEL15":458.0,"ORIGIN_cat":13},"PNS":{"ARR_DEL15":458.0,"ORIGIN_cat":13},"MLU":{"ARR_DEL15":456.0,"ORIGIN_cat":13},"XNA":{"ARR_DEL15":447.0,"ORIGIN_cat":13},"DHN":{"ARR_DEL15":440.0,"ORIGIN_cat":13},"FWA":{"ARR_DEL15":432.0,"ORIGIN_cat":13},"CAK":{"ARR_DEL15":423.0,"ORIGIN_cat":13},"FAY":{"ARR_DEL15":420.0,"ORIGIN_cat":12},"SAN":{"ARR_DEL15":413.0,"ORIGIN_cat":12},"LIT":{"ARR_DEL15":410.0,"ORIGIN_cat":12},"SGF":{"ARR_DEL15":393.0,"ORIGIN_cat":12},"SRQ":{"ARR_DEL15":381.0,"ORIGIN_cat":12},"LFT":{"ARR_DEL15":377.0,"ORIGIN_cat":12},"ECP":{"ARR_DEL15":375.0,"ORIGIN_cat":11},"ILM":{"ARR_DEL15":359.0,"ORIGIN_cat":11},"TUL":{"ARR_DEL15":356.0,"ORIGIN_cat":11},"CSG":{"ARR_DEL15":344.0,"ORIGIN_cat":11},"BUF":{"ARR_DEL15":340.0,"ORIGIN_cat":11},"GPT":{"ARR_DEL15":331.0,"ORIGIN_cat":11},"EYW":{"ARR_DEL15":326.0,"ORIGIN_cat":10},"MLB":{"ARR_DEL15":326.0,"ORIGIN_cat":10},"OMA":{"ARR_DEL15":308.0,"ORIGIN_cat":10},"DAB":{"ARR_DEL15":305.0,"ORIGIN_cat":10},"OAJ":{"ARR_DEL15":303.0,"ORIGIN_cat":10},"OKC":{"ARR_DEL15":303.0,"ORIGIN_cat":10},"AEX":{"ARR_DEL15":300.0,"ORIGIN_cat":10},"GTR":{"ARR_DEL15":291.0,"ORIGIN_cat":9},"MYR":{"ARR_DEL15":287.0,"ORIGIN_cat":9},"BMI":{"ARR_DEL15":286.0,"ORIGIN_cat":9},"MLI":{"ARR_DEL15":274.0,"ORIGIN_cat":9},"BQK":{"ARR_DEL15":269.0,"ORIGIN_cat":9},"GRR":{"ARR_DEL15":266.0,"ORIGIN_cat":9},"CHO":{"ARR_DEL15":266.0,"ORIGIN_cat":9},"EVV":{"ARR_DEL15":265.0,"ORIGIN_cat":8},"DAY":{"ARR_DEL15":260.0,"ORIGIN_cat":8},"PIA":{"ARR_DEL15":259.0,"ORIGIN_cat":8},"SJU":{"ARR_DEL15":252.0,"ORIGIN_cat":8},"ABE":{"ARR_DEL15":250.0,"ORIGIN_cat":8},"VLD":{"ARR_DEL15":242.0,"ORIGIN_cat":8},"PHF":{"ARR_DEL15":238.0,"ORIGIN_cat":8},"PVD":{"ARR_DEL15":234.0,"ORIGIN_cat":7},"ABY":{"ARR_DEL15":233.0,"ORIGIN_cat":7},"ROC":{"ARR_DEL15":222.0,"ORIGIN_cat":7},"SYR":{"ARR_DEL15":214.0,"ORIGIN_cat":7},"MDT":{"ARR_DEL15":200.0,"ORIGIN_cat":7},"ALB":{"ARR_DEL15":196.0,"ORIGIN_cat":7},"ICT":{"ARR_DEL15":191.0,"ORIGIN_cat":6},"PDX":{"ARR_DEL15":187.0,"ORIGIN_cat":6},"CID":{"ARR_DEL15":186.0,"ORIGIN_cat":6},"MSN":{"ARR_DEL15":184.0,"ORIGIN_cat":6},"OAK":{"ARR_DEL15":181.0,"ORIGIN_cat":6},"SNA":{"ARR_DEL15":166.0,"ORIGIN_cat":6},"DSM":{"ARR_DEL15":162.0,"ORIGIN_cat":5},"GRK":{"ARR_DEL15":162.0,"ORIGIN_cat":5},"PWM":{"ARR_DEL15":157.0,"ORIGIN_cat":5},"EWN":{"ARR_DEL15":155.0,"ORIGIN_cat":5},"FNT":{"ARR_DEL15":151.0,"ORIGIN_cat":5},"FSM":{"ARR_DEL15":150.0,"ORIGIN_cat":5},"SJC":{"ARR_DEL15":148.0,"ORIGIN_cat":5},"ELP":{"ARR_DEL15":140.0,"ORIGIN_cat":4},"STT":{"ARR_DEL15":138.0,"ORIGIN_cat":4},"SMF":{"ARR_DEL15":132.0,"ORIGIN_cat":4},"RST":{"ARR_DEL15":124.0,"ORIGIN_cat":4},"TUS":{"ARR_DEL15":113.0,"ORIGIN_cat":4},"ABQ":{"ARR_DEL15":107.0,"ORIGIN_cat":4},"GRB":{"ARR_DEL15":90.0,"ORIGIN_cat":4},"ATW":{"ARR_DEL15":87.0,"ORIGIN_cat":3},"AVP":{"ARR_DEL15":68.0,"ORIGIN_cat":3},"TTN":{"ARR_DEL15":64.0,"ORIGIN_cat":3},"BTV":{"ARR_DEL15":61.0,"ORIGIN_cat":3},"ASE":{"ARR_DEL15":56.0,"ORIGIN_cat":3},"ACY":{"ARR_DEL15":56.0,"ORIGIN_cat":3},"COS":{"ARR_DEL15":56.0,"ORIGIN_cat":3},"MHT":{"ARR_DEL15":50.0,"ORIGIN_cat":2},"LNK":{"ARR_DEL15":48.0,"ORIGIN_cat":2},"HNL":{"ARR_DEL15":46.0,"ORIGIN_cat":2},"FSD":{"ARR_DEL15":42.0,"ORIGIN_cat":2},"HDN":{"ARR_DEL15":34.0,"ORIGIN_cat":2},"JAC":{"ARR_DEL15":32.0,"ORIGIN_cat":2},"ANC":{"ARR_DEL15":31.0,"ORIGIN_cat":1},"BZN":{"ARR_DEL15":26.0,"ORIGIN_cat":1},"EGE":{"ARR_DEL15":23.0,"ORIGIN_cat":1},"STX":{"ARR_DEL15":22.0,"ORIGIN_cat":1},"MTJ":{"ARR_DEL15":10.0,"ORIGIN_cat":1},"FAR":{"ARR_DEL15":7.0,"ORIGIN_cat":1},"TVC":{"ARR_DEL15":7.0,"ORIGIN_cat":1},"FCA":{"ARR_DEL15":5.0,"ORIGIN_cat":0},"ELM":{"ARR_DEL15":5.0,"ORIGIN_cat":0},"RAP":{"ARR_DEL15":4.0,"ORIGIN_cat":0},"RNO":{"ARR_DEL15":3.0,"ORIGIN_cat":0},"SCE":{"ARR_DEL15":2.0,"ORIGIN_cat":0},"MSO":{"ARR_DEL15":0.0,"ORIGIN_cat":0},"GJT":{"ARR_DEL15":0.0,"ORIGIN_cat":0}}'
    origin_json = json.loads(origin_str)
    origin = origin_json[original_origin]['ORIGIN_cat']

    dest_str = '{"ATL":{"ARR_DEL15":96459.0,"DEST_cat":7},"CLT":{"ARR_DEL15":11.0,"DEST_cat":7},"EWR":{"ARR_DEL15":9.0,"DEST_cat":7},"DCA":{"ARR_DEL15":8.0,"DEST_cat":6},"LGA":{"ARR_DEL15":8.0,"DEST_cat":6},"IAH":{"ARR_DEL15":7.0,"DEST_cat":6},"DTW":{"ARR_DEL15":7.0,"DEST_cat":6},"DFW":{"ARR_DEL15":7.0,"DEST_cat":6},"MCO":{"ARR_DEL15":6.0,"DEST_cat":5},"TPA":{"ARR_DEL15":5.0,"DEST_cat":4},"IND":{"ARR_DEL15":5.0,"DEST_cat":4},"ORD":{"ARR_DEL15":5.0,"DEST_cat":4},"BOS":{"ARR_DEL15":5.0,"DEST_cat":4},"DEN":{"ARR_DEL15":5.0,"DEST_cat":4},"MIA":{"ARR_DEL15":3.0,"DEST_cat":3},"LAX":{"ARR_DEL15":3.0,"DEST_cat":3},"PHL":{"ARR_DEL15":3.0,"DEST_cat":3},"JFK":{"ARR_DEL15":3.0,"DEST_cat":3},"PHX":{"ARR_DEL15":3.0,"DEST_cat":3},"BWI":{"ARR_DEL15":3.0,"DEST_cat":3},"AUS":{"ARR_DEL15":3.0,"DEST_cat":3},"PVD":{"ARR_DEL15":2.0,"DEST_cat":2},"ALB":{"ARR_DEL15":2.0,"DEST_cat":2},"RDU":{"ARR_DEL15":2.0,"DEST_cat":2},"RIC":{"ARR_DEL15":2.0,"DEST_cat":2},"SAT":{"ARR_DEL15":2.0,"DEST_cat":2},"SDF":{"ARR_DEL15":2.0,"DEST_cat":2},"OAJ":{"ARR_DEL15":2.0,"DEST_cat":2},"CMH":{"ARR_DEL15":2.0,"DEST_cat":2},"HOU":{"ARR_DEL15":2.0,"DEST_cat":2},"PIA":{"ARR_DEL15":1.0,"DEST_cat":1},"MDW":{"ARR_DEL15":1.0,"DEST_cat":1},"TUS":{"ARR_DEL15":1.0,"DEST_cat":1},"SYR":{"ARR_DEL15":1.0,"DEST_cat":1},"MDT":{"ARR_DEL15":1.0,"DEST_cat":1},"MYR":{"ARR_DEL15":1.0,"DEST_cat":1},"SFO":{"ARR_DEL15":1.0,"DEST_cat":1},"MSP":{"ARR_DEL15":1.0,"DEST_cat":1},"MSY":{"ARR_DEL15":1.0,"DEST_cat":1},"SAV":{"ARR_DEL15":1.0,"DEST_cat":1},"ORF":{"ARR_DEL15":1.0,"DEST_cat":1},"PIT":{"ARR_DEL15":1.0,"DEST_cat":1},"LAS":{"ARR_DEL15":1.0,"DEST_cat":1},"XNA":{"ARR_DEL15":1.0,"DEST_cat":1},"DAL":{"ARR_DEL15":1.0,"DEST_cat":1},"CLE":{"ARR_DEL15":1.0,"DEST_cat":1},"CHO":{"ARR_DEL15":1.0,"DEST_cat":1},"FWA":{"ARR_DEL15":1.0,"DEST_cat":1},"GRR":{"ARR_DEL15":1.0,"DEST_cat":1},"BGR":{"ARR_DEL15":1.0,"DEST_cat":1},"CHS":{"ARR_DEL15":1.0,"DEST_cat":1},"TLH":{"ARR_DEL15":0.0,"DEST_cat":0},"SBN":{"ARR_DEL15":0.0,"DEST_cat":0},"USA":{"ARR_DEL15":0.0,"DEST_cat":0},"PWM":{"ARR_DEL15":0.0,"DEST_cat":0},"CHA":{"ARR_DEL15":0.0,"DEST_cat":0},"CAE":{"ARR_DEL15":0.0,"DEST_cat":0},"ROA":{"ARR_DEL15":0.0,"DEST_cat":0},"SAN":{"ARR_DEL15":0.0,"DEST_cat":0},"BOI":{"ARR_DEL15":0.0,"DEST_cat":0},"BMI":{"ARR_DEL15":0.0,"DEST_cat":0},"TYS":{"ARR_DEL15":0.0,"DEST_cat":0},"AVP":{"ARR_DEL15":0.0,"DEST_cat":0},"AGS":{"ARR_DEL15":0.0,"DEST_cat":0},"TTN":{"ARR_DEL15":0.0,"DEST_cat":0},"AVL":{"ARR_DEL15":0.0,"DEST_cat":0},"BDL":{"ARR_DEL15":0.0,"DEST_cat":0},"SGF":{"ARR_DEL15":0.0,"DEST_cat":0},"SLC":{"ARR_DEL15":0.0,"DEST_cat":0},"SMF":{"ARR_DEL15":0.0,"DEST_cat":0},"STL":{"ARR_DEL15":0.0,"DEST_cat":0},"SEA":{"ARR_DEL15":0.0,"DEST_cat":0},"KOA":{"ARR_DEL15":0.0,"DEST_cat":0},"CVG":{"ARR_DEL15":0.0,"DEST_cat":0},"JAX":{"ARR_DEL15":0.0,"DEST_cat":0},"JAN":{"ARR_DEL15":0.0,"DEST_cat":0},"LCK":{"ARR_DEL15":0.0,"DEST_cat":0},"LEX":{"ARR_DEL15":0.0,"DEST_cat":0},"ACY":{"ARR_DEL15":0.0,"DEST_cat":0},"LIH":{"ARR_DEL15":0.0,"DEST_cat":0},"LIT":{"ARR_DEL15":0.0,"DEST_cat":0},"LYH":{"ARR_DEL15":0.0,"DEST_cat":0},"MCI":{"ARR_DEL15":0.0,"DEST_cat":0},"IAD":{"ARR_DEL15":0.0,"DEST_cat":0},"HNL":{"ARR_DEL15":0.0,"DEST_cat":0},"HHH":{"ARR_DEL15":0.0,"DEST_cat":0},"GSP":{"ARR_DEL15":0.0,"DEST_cat":0},"MKE":{"ARR_DEL15":0.0,"DEST_cat":0},"GSO":{"ARR_DEL15":0.0,"DEST_cat":0},"FLL":{"ARR_DEL15":0.0,"DEST_cat":0},"OAK":{"ARR_DEL15":0.0,"DEST_cat":0},"OKC":{"ARR_DEL15":0.0,"DEST_cat":0},"OMA":{"ARR_DEL15":0.0,"DEST_cat":0},"DSM":{"ARR_DEL15":0.0,"DEST_cat":0},"DAY":{"ARR_DEL15":0.0,"DEST_cat":0},"PBI":{"ARR_DEL15":0.0,"DEST_cat":0},"PDX":{"ARR_DEL15":0.0,"DEST_cat":0},"PHF":{"ARR_DEL15":0.0,"DEST_cat":0},"ABE":{"ARR_DEL15":0.0,"DEST_cat":0}}'
    dest_json = json.loads(dest_str)
    dest = dest_json[destination]['DEST_cat']

    # try:
    url='http://api.weatherapi.com/v1/forecast.json?key=385cf586f9c5428b85d201414211009&q={city}&days=3&aqi=no&alerts=no'.format(city=original_origin)
    response=requests.get(url)
    #response.raise_for_status()
    JsonResponse=response.json()
    # print(JsonResponse)
    # hours = JsonResponse['forecast']['forecastday'][0]['hour']
    day1 = JsonResponse['forecast']['forecastday'][0]['date']
    day2 = JsonResponse['forecast']['forecastday'][1]['date']
    day3 = JsonResponse['forecast']['forecastday'][2]['date']

    day_matching = departure_time.split(" ")[0]

    hours = []
    if day_matching == day1:
        hours = JsonResponse['forecast']['forecastday'][0]['hour']

    elif day_matching == day2:
        hours = JsonResponse['forecast']['forecastday'][1]['hour']
        
    elif day_matching == day3:
        hours = JsonResponse['forecast']['forecastday'][2]['hour']

    for hour in hours:
        print(hour["time"])
        
        if departure_time==hour["time"]:
            print(departure_time)
            datetime_object = datetime.strptime(departure_time, '%Y-%m-%d %H:%M')
            week_day = datetime_object.weekday()
            if week_day in [5,6]:
                weekend=1
            else:
                weekend=0

            hourly_time = datetime_object.strftime("%H:%M")
            print("time:", hourly_time)
            hourly_time = hourly_time.replace(":","")

            temperature=hour["temp_f"]
            dew_point=hour['dewpoint_f']
            humidity=hour['humidity']
            wind_speed=hour['wind_kph']
            pressure=hour['pressure_mb']
            precipitation=hour['precip_mm']
            print(temperature)
            print(dew_point)
            print(humidity)
            print(wind_speed)
            print(pressure)
            print(precipitation)
            
            # sample = np.array([[ weekend, airline, hourly_time, origin, dest, temperature, dew_point, humidity, wind_speed, pressure, precipitation]])  # randomly choose a sample from the test set
            sample = [ weekend, airline, hourly_time, origin, dest, temperature, dew_point, humidity, wind_speed, pressure, precipitation]
            print(sample)
            
            sample = np.array(sample).reshape(1, 11)
            X_test_scaled = scaler.transform(sample)
            X_test_scaled = minmax.transform(X_test_scaled)
            print(X_test_scaled)

            prediction = model.predict(X_test_scaled)

            return jsonify({"Delay": str(prediction[0])})
    else:
        return jsonify({"Error": str("Time not matched")})

if __name__=="__main__":
    app.run(host= '0.0.0.0')
