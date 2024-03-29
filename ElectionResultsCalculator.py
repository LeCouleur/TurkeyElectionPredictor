import numpy as np

# 1.Orc 2.Optimar 3.Mak 4.Piar 5.themis 6.avrasya
# 1.akp 2.mhp 3.Mak 4.Piar 5.themis 6.avrasya
election1_polls = np.array([
      
    [28.1, 6.3, 1.6, 1.2, 28.3, 11.7, 2.0, 2.5, 0.2, 0.2, 9.3, 1.5, 1.5, 2.3],
    [38.4, 7.7, 0.9, 1, 25.9, 7.8, 0.3, 0.3, 0.3, 0.1, 9.5, 0.7, 0.3, 6.2],
    [34.2, 6.3, 0.4, 1.3, 26, 13.6, 2.1, 1, 0.9, 0.4, 9.3, 0.3, 1, 1],
    [30.8, 7.1, 0.2, 0.8, 32.3, 8.3, 2.1, 3.7, 0.4, 0.1, 11.6, 1, 1, 1.4],
    [27.5, 3.6, 0.2, 2.2, 30.8, 9.7, 2.4, 1.1, 1.1, 0.1, 9.7, 3.1, 8.5, 2.3],
    [29.1, 7.0, 0.1, 1.5, 31.9, 10.5, 2.1, 0.8, 0.9, 0.5, 10.2, 1.6, 1, 1.6]
])
election1_names = np.array([
    ['AKP', 'MHP', 'BBP', 'YRP', 'CHP', 'İYİ', 'DEVA', 'GP', 'SP', 'DP', 'HDP', 'TİP', 'ZP', 'MP' ]
])


weights = np.array([0.57642149, 1.45889644, -1.06019027, 1.13938594, 0.99448702, 0.98536125])



election1_predictions = np.dot(election1_polls.T, weights)
scale_factor = 100 / np.sum(election1_predictions)
election1_predictions *= scale_factor

print("Results:")
for i in range(len(election1_names[0])):
    print(election1_names[0][i] + ": " + str(round(election1_predictions[i], 2)) + "%")
