import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV correctamente
file_path = r'C:\Users\Josema\Desktop\Alzheimer-detection-project\ADNI1.csv'  # Asegúrate de que este es el camino correcto al archivo CSV
data = pd.read_csv(file_path, delimiter=',')

# Separar los valores de la primera columna
separated_values = data.iloc[:, 0].str.split(',', expand=True)

# Extraer las columnas 'Subject' (posición 1) y 'Sex' (posición 3)
subjects = separated_values[1]
sexes = separated_values[3]

# Eliminar duplicados basados en 'Subject' para contar cada usuario una sola vez
unique_subjects = separated_values.drop_duplicates(subset=1)

# Contar los valores de 'Sex'
sex_counts = unique_subjects[3].value_counts()

# Renombrar los valores de la columna 'Sex' para la gráfica
sex_counts.index = ['Male' if sex == 'M' else 'Female' for sex in sex_counts.index]

# Crear una gráfica circular de la distribución de sexos
plt.subplot(1, 2, 2)
sex_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'], labels=sex_counts.index)
plt.title('Distribution of Sexes (Unique Subjects)')
plt.ylabel('')  # Ocultar la etiqueta del eje y

# Mostrar las gráficas
plt.tight_layout()
plt.show()




