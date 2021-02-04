import contextlib

# 인스턴스화 하지 않고 '클래스' 상태로 이용
# 인스턴스는 여러개 생성할 수 있지만 클래스는 항상 하나만 존재하기 때문
class Config:
    enable_backprop = True


@contextlib.contextmanager
# name -> Config 속성의 이름(클래스 속성 이름), str
def using_config(name, value):
    # name을 getattr 함수에 넘겨 Config 클래스에서 꺼내오기
    old_val = getattr(Config, name)
    # 새로운 값을 설정
    setattr(Config, name, value)
    try:
        yield
    finally:
        # with 블록을 빠져나오면서 다시 원래 값(old_val)으로 복원
        setattr(Config, name, old_val)