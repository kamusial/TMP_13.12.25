class Auto:
    def __init__(self, barwa, paliwo, wiek):
        self.kolor = barwa
        self.paliwo = paliwo
        self.rocznik = 2025 - wiek
        self.kondycja = 5
        self.tryb_ekonomiczny = False
        self.spalanie_na_100 = 14
        self.mandaty = []
        self.komentarze = []

    def zasieg(self):
        zasieg = self.paliwo / self.spalanie_na_100 * 100
        return round(zasieg * 0.9)

    def ustaw_tryb(self, tryb):
        if tryb == 'eco':
            self.spalanie_na_100 = 10
            self.tryb_ekonomiczny = True
            print('Tryb eco')
        elif tryb == 'normal':
            self.spalanie_na_100 = 14
            self.tryb_ekonomiczny = False
            print('Tryb normal')
        else:
            print('tryb nierozpoznany, brak zmian')

    def add_comment(self, comment):
        self.komentarze.append(comment)

    def read_comments(self):
        passwd = input('Podaj haslo: ')
        if passwd == '1234':
            print(self.komentarze)




moje_auto = Auto('blue', 7, 12)