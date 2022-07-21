from hanspell import spell_checker
#from pykospacing import Spacing

# sort, unique -> 리눅스 창에서 하는 게 나음
# 특수문자는 감정과 관련될 수 있어서 제거하지 않음

# basic 전처리
def basic_preprocess(sentence):
    # 기호 일반화
    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }
    for p in punct_mapping:
        sentence = sentence.replace(p, punct_mapping[p])
    new_sentence = sentence.lower() # 대소문자 통일
    new_sentence = new_sentence.strip() # 좌우 공백 제거
    return new_sentence

# 맞춤법 교정
def spell_check(sentence):
    result = spell_checker.check(sentence)
    new_sentence = result.as_dict()['checked']
    return new_sentence

# 띄어쓰기 교정
#def spacing_check(sentence):
#    spacing = Spacing()
#    new_sentence = spacing(sentence) 
#    return new_sentence

# 이상한 문자 제거


# 전체 전처리 함수
def preprocess(sentence):
    # 기본 전처리
    new_sentence = basic_preprocess(sentence)

    # 맞춤법
    #new_sentence = spell_check(new_sentence)

    # 띄어쓰기
    #new_sentence = spacing_check(new_sentence)

    return new_sentence

