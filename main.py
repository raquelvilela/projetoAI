import numpy as np
import cv2
import os
from skimage.feature import hog
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Diretórios das imagens
nike_dir = 'C:/dev/pythonprojects/projetoAI/img/nike_images'
other_dir = 'C:/dev/pythonprojects/projetoAI/img/others'
sneakers_dir = 'C:/dev/pythonprojects/projetoAI/img/sneakers'
not_sneakers_dir = 'C:/dev/pythonprojects/projetoAI/img/not_sneakers'

# Função para carregar e processar imagens
def load_images_from_folder(folder, label, img_size=(64, 64)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                hog_features = hog(img_gray, orientations=9, pixels_per_cell=(8, 8),
                                   cells_per_block=(2, 2), block_norm='L2-Hys')
                images.append(hog_features)
                labels.append(label)
        except Exception as e:
            print(f"Erro ao carregar a imagem {img_path}: {e}")
    return np.array(images), np.array(labels)

# Carregar e processar as imagens
nike_images, nike_labels = load_images_from_folder(nike_dir, 1)
other_images, other_labels = load_images_from_folder(other_dir, 0)
sneakers_images, sneakers_labels = load_images_from_folder(sneakers_dir, 1)
not_sneakers_images, not_sneakers_labels = load_images_from_folder(not_sneakers_dir, 0)

# Verificar se os dados estão balanceados
print(f"Número de imagens Nike: {len(nike_images)}")
print(f"Número de imagens de outras marcas: {len(other_images)}")
print(f"Número de imagens de tênis: {len(sneakers_images)}")
print(f"Número de imagens de não-tênis: {len(not_sneakers_images)}")

# Combinar e dividir os dados para classificadores
# Classificador Tênis vs. Não-Tênis
X_sneakers = np.concatenate((sneakers_images, not_sneakers_images), axis=0)
y_sneakers = np.concatenate((sneakers_labels, not_sneakers_labels), axis=0)

# Classificador Nike vs. Não-Nike
X_nike = np.concatenate((nike_images, other_images), axis=0)
y_nike = np.concatenate((nike_labels, other_labels), axis=0)

# Dividir os dados em treino e teste para classificadores
X_sneakers_train, X_sneakers_test, y_sneakers_train, y_sneakers_test = train_test_split(X_sneakers, y_sneakers, test_size=0.2, random_state=42)
X_nike_train, X_nike_test, y_nike_train, y_nike_test = train_test_split(X_nike, y_nike, test_size=0.2, random_state=42)

# Treinar o modelo Random Forest para tênis vs. não-tênis
sneakers_model = RandomForestClassifier(n_estimators=100, random_state=42)
sneakers_model.fit(X_sneakers_train, y_sneakers_train)

# Treinar o modelo Random Forest para Nike vs. não-Nike
nike_model = RandomForestClassifier(n_estimators=100, random_state=42)
nike_model.fit(X_nike_train, y_nike_train)

# Fazer previsões no conjunto de teste e avaliar a precisão para tênis vs. não-tênis
y_sneakers_pred = sneakers_model.predict(X_sneakers_test)
sneakers_accuracy = accuracy_score(y_sneakers_test, y_sneakers_pred)
print(f"Acurácia no conjunto de teste (Tênis vs. Não-Tênis): {sneakers_accuracy * 100:.2f}%")
print("Relatório de Classificação (Tênis vs. Não-Tênis):")
print(classification_report(y_sneakers_test, y_sneakers_pred, zero_division=0))
print("Matriz de Confusão (Tênis vs. Não-Tênis):")
print(confusion_matrix(y_sneakers_test, y_sneakers_pred))

# Fazer previsões no conjunto de teste e avaliar a precisão para Nike vs. não-Nike
y_nike_pred = nike_model.predict(X_nike_test)
nike_accuracy = accuracy_score(y_nike_test, y_nike_pred)
print(f"Acurácia no conjunto de teste (Nike vs. Não-Nike): {nike_accuracy * 100:.2f}%")
print("Relatório de Classificação (Nike vs. Não-Nike):")
print(classification_report(y_nike_test, y_nike_pred, zero_division=0))
print("Matriz de Confusão (Nike vs. Não-Nike):")
print(confusion_matrix(y_nike_test, y_nike_pred))

# Verificação de Imagem
def verify_image(image_path, sneakers_model, nike_model, img_size=(64, 64)):
    try:
        img = cv2.imread(image_path)
        if img is not None:
            original_img = img.copy()  # Preserve the original image
            img = cv2.resize(img, img_size)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            hog_features = hog(img_gray, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), block_norm='L2-Hys')
            is_sneakers = sneakers_model.predict([hog_features])[0]
            if is_sneakers == 1:
                is_nike = nike_model.predict([hog_features])[0]
                if is_nike == 1:
                    return "É um tênis Nike!", original_img
                else:
                    return "É um tênis, mas não é da Nike.", original_img
            else:
                return "Não é um tênis.", original_img
        else:
            raise ValueError("Imagem não encontrada ou inválida")
    except Exception as e:
        print(f"Erro ao processar a imagem {image_path}: {e}")
        return "Erro ao processar a imagem.", None

# Exemplo de uso
image_path = 'C:/dev/pythonprojects/projetoAI/img/bone.png'  # Caminho tênis a comparar
result_text, original_img = verify_image(image_path, sneakers_model, nike_model)

# Visualizar Resultado
def plot_result(img, result_text):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
    plt.title(result_text)
    plt.axis('off')
    plt.show()

# Visualizar o resultado
if original_img is not None:
    plot_result(original_img, result_text)
else:
    print(result_text)
