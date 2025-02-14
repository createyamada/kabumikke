from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import timedelta, datetime
from pydantic import BaseModel
from typing import Optional
import pytz

router = APIRouter()

# JWT設定
SECRET_KEY = "SECRET_KEY"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# パスワードのハッシュ化に使用するための設定
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 仮のデータベースデータ
fake_users_db = {
    "testuser":{
        "username": "testuser",
        "hashed_password": "$2b$12$o.F1CWNLwlxSB.lb9wY5IOduWEdXUj0F6bAGVJw2kuhQwydEuekpu" # testpassword 
    }
}

# userモデル
class User(BaseModel):
    username: str
    hashed_password: Optional[str] = None

# パスワードを検証する関数
def verify_password(plain_password, hashed_password):
    print('pass')
    return pwd_context.verify(plain_password, hashed_password)

# ユーザを取得する関数
def get_user(username: str):
    if username in fake_users_db:
        user_dict = fake_users_db[username]
        return User(**user_dict)

# パスワードが正しいかどうかをチェックする関数
def authenticate_user(username: str, password: str):
    user = get_user(username)
    # パスワードをハッシュ化
    newpass = pwd_context.hash(password)
    print('newpass')
    print(newpass)
    if not user:
        return False
    if not verify_password(password,user.hashed_password):
        return False
    print('test')
    return user

# トークンをを作成する関数
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    japan_tz = pytz.timezone('Asia/Tokyo')
    if expires_delta:
        expire = datetime.utcnow().astimezone(japan_tz) + expires_delta
    else:
        expire = datetime.utcnow().astimezone(japan_tz) + timedelta(minutes=15)
    to_encode.update({"exp":expire})
    expire = expire.strftime('%Y-%m-%d %H:%M:%S')
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return { "access_token": encoded_jwt , "token_expire" : expire }

# OAuth2パスワードベアラースキームの設定
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# トークンを受け入れるエンドポイント
# curl -X POST http://localhost:8888/api/auth/token -H 'Content-Type: application/x-www-form-urlencoded' -d "username=testuser&password=testpassword"
@router.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers = {"WWW-Authenticate": "Bearer"}
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub":user.username}, expires_delta = access_token_expires
    )
    return {"token_data": access_token , "token_type": "bearer"}

# プロファイル情報を取得するエンドポイント
# curl -X GET http://localhost:8888/api/auth/profile -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0dXNlciIsImV4cCI6MTcwOTY5NDA4NH0.9Rhkf0JICKOO7YyjePXWtbTSU7jjuKTwJ4oBPQMb-yE"
@router.get("/profile")
async def get_profile(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code = status.HTTP_401_UNAUTHORIZED,
                detail = "Could not validate credentials",
                headers = {"WWW-Authenticate": "Bearer"}
            )
        return fake_users_db[username]
    except JWTError:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Could not validate credentials",
            headers = {"WWW-Authenticate": "Bearer"} 
        )