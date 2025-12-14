employee = ['Jan' ,'Kowalski', 'male', 26]

try:
    age = int(input('Ile masz lat?   '))
    seniority = int(input('Ile czasu pracujesz w firmie X?   '))
    points = float(input('Ile masz punktów?   '))
    holiday_request = int(input('Ile chcesz wziąć dni urlopu?   '))
    potential = seniority / age + points / seniority
    if holiday_request <= employee[3]:
        print('Dostaniesz urlop')
        if potential > 5:
            print('Dostaniesz dodatkowy dzień urlopu')
    else:
        print('nie masz tyle urlopu')
    print(f'Twój potencjał to: {potential}')
except IndexError:
    print('Błąd, odnosisz się dodanych, których nie ma')
    print('Nie wiem, ile masz urlopu')
except ZeroDivisionError:
    if holiday_request <= employee[3]:
        print('Dostaniesz urlop')
    else:
        print('nie masz tyle urlopu')
except ValueError:
    print('Błąd - zły typ danych - nie policzę')


print('koniec programu')