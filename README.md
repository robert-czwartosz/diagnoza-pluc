Temat projektu
===========

Diagnoza zapalenia płuc na podstawie skanu rentgenowskiego
Współautor: Daniel Tarnawski

Cele projektu 
==========

Cele projektu:
* Implementacja i porównanie metod diagnozy zapalenia płuc na podstawie zdjęcia rentgenowskiego
* Implementacja aplikacji do diagnozy zapalenia płuc na podstawie wybranego zdjęcia rentgenowskiego

Wymagania
==========

Wymagania: Python 3.7 + numpy, opencv, scikit-image, scikit-learn, joblib, mahotas, 
						tensorflow-gpu

Funkcjonalności:
==============
1. Możliwość porównania trzech metod wykrywania zapalenia płuc(macierz współwystąpień + sieć MLP, momenty Zernikego + sieć MLP oraz sieć konwolucyjna).

2. Możliwość diagnozy zapalenia płuc na podstawie wybranego zdjęcia rentgenowskiego.

Efektywność metod diagnozy zapalenia płuc.
==================================
Dokładność metod badana na zbiorze testowym:
1. Momenty Zernikego + sieć MLP - 81%
2. Macierz współwystąpień + sieć MLP - 50%
3. Sieć konwolucyjna - 85%

Druga metoda osiągnęła bardzo niską dokładność(wszystkie zdjęcia zostały zakwalifikowane do jednej kategorii). Możliwą przyczyną tego są błędy w algorytmie lub zły model sieci neuronowej.

Screeny
========
Po uruchomieniu aplikacji interfejs użytkownika wygląda następująco:
![Interfejs po uruchomieniu](screeny/screen1.png "Interfejs po uruchomieniu aplikacji")
Po kliknięciu przycisku **Wczytaj** i wybraniu pliku ze zdjęciem w oknie pojawia się wybrane zdjęcie.
![Wczytane zdjęcie](screeny/screen2.png "Wczytane zdjęcie")
Po kliknięciu przycisku **Zbadaj** otrzymano wynik diagnozy.
![Wynik diagnozy](screeny/screen3.png "Wynik diagnozy")

Konfiguracja projektu
===================

1. Pobierz dane ze strony [https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) i wypakuj je do folderu **chest-xray-pneumonia**.
1. Uruchom polecenie **python macierz.py**, aby przetestować działanie metody stosującej macierz współwystąpień i sieć MLP.
2. Uruchom polecenie **python zernik.py**, aby przetestować działanie metody stosującej momenty Zernikego i sieć MLP.
3. Uruchom polecenie **python diagnozaTensorflow.py**, aby przetestować działanie metody stosującej konwolucyjną sieć neuronową oraz wytrenować model używany przez program **diagnozaTensorflowGUI.py**.
4. Uruchom polecenie **python diagnozaTensorflow.py**, aby uzyskać prawdopodobieństwo zapalenia płuc na podstawie wybranego zdjącia rentgenowskiego.


TODO:
====

* refaktoryzacja kodu
* poszukanie ewentualnego błędu w drugiej metodzie
* udoskonalenie interfejsu użytkownika
* dokumentacja