import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

data = np.load('prices_new_test.npz')
true_prices = data['true'].ravel()
pred_prices = data['pred'].ravel()

mean_price = np.mean(true_prices)
print("Mean price:", mean_price)
dummy_pred = np.full(pred_prices.shape[0], mean_price)
# pred_prices = dummy_pred

# top_10 = np.sort(pred_prices, axis=0)
# print("10 highest predictions:", top_10)

# top_10 = np.sort(true_prices, axis=0)
# print("10 highest true:", top_10)

abs_errors = np.abs(pred_prices - true_prices)

print("MAE:", np.mean(abs_errors))
print("MAPE:", np.mean(abs((true_prices - pred_prices) / true_prices)) * 100)
# print(np.median(abs_errors))
print("R2 score:", r2_score(true_prices, pred_prices))

plt.figure(figsize=(8, 6))
plt.scatter(true_prices, abs_errors, alpha=0.5)
plt.xlabel("Resnična cena")
plt.ylabel("Absolutna napaka")
plt.title("Prikaz absolutnih napak napovedi")
plt.grid(True)
plt.tight_layout()
plt.savefig("testing.png")
plt.show()

# print("Some predictions:")
# print(pred_prices[100:105])
# print(true_prices[100:105])

# mae = np.mean(abs(true_prices - pred_prices))
# rmse = np.sqrt(np.mean((true_prices - pred_prices) ** 2))
# mape = np.mean(abs((true_prices - pred_prices) / true_prices)) * 100

# print("MAE:", mae)
# print("RMSE:", rmse)
# print("MAPE:", mape)