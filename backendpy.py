import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap

# Dane dotyczące cech frameworków
dane_frameworkow = {
    'Framework': ['Laravel', 'Symfony', 'Laminas', 'CodeIgniter'],
    'Wydajność': [2, 3, 2, 1],
    'Dokumentacja': [3, 3, 3, 2],
    'Społeczność': [3, 3, 2, 1],
    'Bezpieczeństwo': [3, 3, 2, 1],
    'Popularność': [3, 2, 1, 2],
    'Trudność nauki': [2, 2, 3, 1],
    'Złożoność': [2, 3, 3, 2],
    'Wykorzystanie zasobów': [2, 2, 3, 1],
}

# Konwersja danych na DataFrame
frameworks_df = pd.DataFrame(dane_frameworkow)

# Definiowanie cech korzyści i kosztów
cechy_korzysci = ['Wydajność', 'Dokumentacja',
                  'Społeczność', 'Bezpieczeństwo', 'Popularność']
cechy_kosztow = ['Trudność nauki', 'Złożoność', 'Wykorzystanie zasobów']

# Definiowanie wag (w tym przypadku równe dla wszystkich cech)
wagi = {
    'Wydajność': 3,
    'Dokumentacja': 1,
    'Społeczność': 1,
    'Bezpieczeństwo': 3,
    'Popularność': 1,
    'Trudność nauki': 2,
    'Złożoność': 1,
    'Wykorzystanie zasobów': 1,
}

# Konstrukcja macierzy X
X = frameworks_df[cechy_korzysci + cechy_kosztow].values

# Normalizacja wartości cech korzyści
znormalizowane_X_korzysci = X[:, :len(
    cechy_korzysci)] / np.sqrt((X[:, :len(cechy_korzysci)] ** 2).sum(axis=0))

# Przeskalowanie wartości cech kosztów
X_inverted_koszty = 1 / X[:, len(cechy_korzysci):]

# Połączenie znormalizowanych wartości cech korzyści i przeskalowanych cech kosztów
znormalizowane_X = np.hstack((znormalizowane_X_korzysci, X_inverted_koszty))

# Obliczanie ważonych znormalizowanych wartości
wektor_wag = np.array(list(wagi.values()))
ważone_znormalizowane_X = znormalizowane_X * wektor_wag

# Konstrukcja wariantów idealnych
pozytywne_rozwiazanie_idealne = np.max(ważone_znormalizowane_X, axis=0)
negatywne_rozwiazanie_idealne = np.min(ważone_znormalizowane_X, axis=0)

# Obliczanie odległości do wariantów idealnych
odległość_do_pozytywnego_idealnego = np.sqrt(
    np.sum((ważone_znormalizowane_X - pozytywne_rozwiazanie_idealne) ** 2, axis=1))
odległość_do_negatywnego_idealnego = np.sqrt(
    np.sum((ważone_znormalizowane_X - negatywne_rozwiazanie_idealne) ** 2, axis=1))

# Obliczanie całkowitego wyniku
całkowity_wynik_preferencyjny = odległość_do_negatywnego_idealnego / \
    (odległość_do_pozytywnego_idealnego + odległość_do_negatywnego_idealnego)

# Sortowanie indeksów według całkowitego wyniku
indeksy = np.argsort(całkowity_wynik_preferencyjny)

# Wizualizacja alternatyw

plt.figure(figsize=(8, 6))
plt.scatter(odległość_do_pozytywnego_idealnego, odległość_do_negatywnego_idealnego,
            c=całkowity_wynik_preferencyjny, cmap='viridis')
plt.xlabel('Odległość do pozytywnego rozwiązania idealnego')
plt.ylabel('Odległość do negatywnego rozwiązania idealnego')

plt.colorbar(label='Całkowity Wynik Preferencyjny')
for i in range(len(frameworks_df)):
    plt.text(odległość_do_pozytywnego_idealnego[i],
             odległość_do_negatywnego_idealnego[i], frameworks_df['Framework'][i])
plt.grid(True)
plt.show()

# Ustalanie indeksu najlepszej alternatywy
najlepszy_indeks_alternatywy = indeksy[-1]

# Wykres słupkowy z wynikami
plt.figure(figsize=(6, 2))
frameworky = frameworks_df['Framework']
wyniki = całkowity_wynik_preferencyjny[indeksy]

colors = plt.cm.viridis(
    całkowity_wynik_preferencyjny[indeksy] / max(całkowity_wynik_preferencyjny))

plt.barh(frameworky[indeksy], wyniki, color=colors,
         height=0.4)
plt.xlabel('Całkowity Wynik Preferencyjny')
plt.grid(False)

plt.yticks(range(len(frameworky)), frameworky[indeksy])

for index, value in enumerate(wyniki):
    plt.text(value, index, f'{value:.2f}', va='center',
             ha='left', fontsize=8, color='black')

plt.show()

# Definiowanie cech korzyści i kosztów
cechy = cechy_korzysci + cechy_kosztow
N = len(cechy)

# Indeksy dla cech
x = np.arange(N)

fig, ax = plt.subplots(figsize=(10, 6))

# Porównanie cech frameworków bez uwzględnienia wag
for i in range(len(frameworky)):
    values = frameworks_df.iloc[i][cechy].values
    ax.plot(x, values, marker='o', linestyle='-', label=frameworky[i])

ax.set_xticks(x)
ax.set_xticklabels(cechy)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Cechy')
plt.ylabel('Wartości cech')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.grid(True)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))


frameworks_df_inverted = frameworks_df.copy()
frameworks_df_inverted[cechy_kosztow] = 5 - \
    frameworks_df_inverted[cechy_kosztow]


for i in range(len(frameworky)):
    values = frameworks_df_inverted.iloc[i][cechy].values * \
        np.array([wagi[c] for c in cechy])
    ax.plot(x, values, marker='o', linestyle='-', label=frameworky[i])

# 'Porównanie cech frameworków z uwzględnieniem wag i cech kosztów'
ax.set_xticks(x)
ax.set_xticklabels(cechy)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Cechy')
plt.ylabel('Wartości cech (ważone)')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.grid(True)
plt.tight_layout()
plt.show()

# Wizualizacja macierzy decyzyjnej w tabeli
fig, ax = plt.subplots(figsize=(13, 6))
ax.axis('tight')
ax.axis('off')


def wrap_text(text, width):
    return "\n".join(textwrap.wrap(text, width))


headers = [wrap_text(header, 14) for header in frameworks_df.columns]
cell_text = frameworks_df.apply(lambda x: x.map(
    lambda y: wrap_text(str(y), 12))).values
decision_matrix_table = ax.table(
    cellText=cell_text, colLabels=headers, cellLoc='center', loc='center')

# Macierz Decyzyjna Frameworków
for (i, j), cell in decision_matrix_table.get_celld().items():
    if i == 0:
        cell.set_facecolor('#f2f2f2')
        cell.set_text_props(weight='bold')
    else:
        cell.set_facecolor('white')
    cell.set_edgecolor('black')
    cell.set_text_props(ha='center', va='center')
    cell.set_height(0.1)
    cell.set_width(0.1)
plt.show()
