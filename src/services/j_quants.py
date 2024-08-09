# import requests
# import json
# # from fastapi import APIRouter, Depends, HTTPException, status
# from services import config
# from datetime import datetime

# # class SessionData:
# #     def __init__(self):
# #         self.data = {}

# # session_data = SessionData()

# # def get_session_data():
# #     return session_data

# # @router.get("/refresh_token")
# # curl -X GET http://localhost:8888/api/call_j_quants/refresh_token -H 'Content-Type: application/json'
# async def get_refresh_token():
#     """
#     ルンゲクッタ法による常微分方程式の数値解法

#     Parameters:
#     - f:object 速度を更新する力{"x":vxの力,"y":vyの力}
#     - vx: x軸初速度
#     - vy: y軸初速度
#     - dt:array 刻み幅リスト

#     Returns:
#     - result: x,y軸の位置ベクトルのリスト[{"x":0,"y":0},...]
#     """


#     #　環境変数からJ_QUANTS認証情報取得
#     id = config.J_QUANTS_ID
#     password = config.J_QUANTS_PASS
    
#     # リフレッシュIDを取得するリクエスト
#     params = {
#         "mailaddress":id,
#         "password":password
#     }

#     result = requests.post("https://api.jquants.com/v1/token/auth_user", data=json.dumps(params))
#     # 取得したリフレッシュトークンをグローバル変数に代入
#     return result.json()['refreshToken']
    

# # @router.get("/id_token")
# # curl -X GET http://localhost:8888/api/call_j_quants/id_token -H 'Content-Type: application/json'
# async def get_id_token(refresh_token: str):
#     """
#     ルンゲクッタ法による常微分方程式の数値解法

#     Parameters:
#     - f:object 速度を更新する力{"x":vxの力,"y":vyの力}
#     - vx: x軸初速度
#     - vy: y軸初速度
#     - dt:array 刻み幅リスト

#     Returns:
#     - result: x,y軸の位置ベクトルのリスト[{"x":0,"y":0},...]
#     """


#     result = requests.post(f"https://api.jquants.com/v1/token/auth_refresh?refreshtoken={refresh_token}")
#     return result.json()['idToken']


# # @router.get("/get_info")
# # curl -X GET http://localhost:8888/api/call_j_quants/get_info -H 'Content-Type: application/json'
# async def get_info(code:str,id_token:str):
#     # code = '3633'
#     # id_token = session_data.data['id_token']
#     headers = {'Authorization': 'Bearer {}'.format(id_token)}
#     result = requests.get(f"https://api.jquants.com/v1/listed/info?code={code}", headers=headers)
#     print(result.json())

# # @router.get("/get_value")
# # curl -X GET http://localhost:8888/api/call_j_quants/get_value -H 'Content-Type: application/json'
# async def get_data(code_:str,id_token:str):
#     """
#     ルンゲクッタ法による常微分方程式の数値解法

#     Parameters:
#     - f:object 速度を更新する力{"x":vxの力,"y":vyの力}
#     - vx: x軸初速度
#     - vy: y軸初速度
#     - dt:array 刻み幅リスト

#     Returns:
#     - result: x,y軸の位置ベクトルのリスト[{"x":0,"y":0},...]
#     """

#     # 銘柄コード、期間を指定
#     code_ = "8697" 
#     from_ = '2022-09-20'
#     to_ = datetime.now().strftime('%Y-%m-%d')

#     # id_token = session_data.data['id_token']
#     headers = {'Authorization': 'Bearer {}'.format(id_token)}

#     # 株価四本値の取得
#     prices = requests.get(f"https://api.jquants.com/v1/prices/daily_quotes?code={code_}", headers=headers)
#     # prices = requests.get(f"https://api.jquants.com/v1/prices/daily_quotes?code={code_}&from={from_}&to={to_}", headers=headers)

#     # 財務情報の取得
#     statements = requests.get(f"https://api.jquants.com/v1/fins/statements?code={code_}", headers=headers)

#     return {
#         'prices':prices.json(),
#         'statements':statements.json()
#     }

