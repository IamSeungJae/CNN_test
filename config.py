import json

try:
    with open("config.json", encoding='utf-8') as f:
        config_data = f.read().strip()  # 공백 제거
        if not config_data:  # 파일이 비어있다면 예외 발생
            raise ValueError("config.json 파일이 비어 있습니다.")

        config = json.loads(config_data)

except json.JSONDecodeError:
    print("JSON 파일 형식이 올바르지 않습니다. 파일 내용을 확인하세요.")
    exit()
except FileNotFoundError:
    print("config.json 파일이 존재하지 않습니다. 파일을 생성하세요.")
    exit()
except ValueError as e:
    print(e)
    exit()
