age = input('Ile masz lat?   ')

try:
    age = int(age)
    print(f'Za 10 lat będziesz miał {age + 10}')
except:
    print('Nie da się')


print('koniec programu')