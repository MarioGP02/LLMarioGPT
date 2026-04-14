import requests

# URL del Quijote (texto plano)
url = "https://www.gutenberg.org/cache/epub/2000/pg2000.txt"

# Añadimos un "User-Agent" para que el servidor crea que somos un navegador
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

print("Reading Cervantes... 📖")

try:
    response = requests.get(url, headers=headers)
    response.raise_for_status() # Lanza error si la descarga falla
    texto = response.text

    if len(texto) < 1000:
        print("❌ Error: El texto descargado es demasiado corto. Algo falló.")
    else:
        # Buscamos el inicio real de la novela para quitar el prólogo legal de Gutenberg
        # Si no lo encuentra, usará el texto completo
        inicio_idx = texto.find("I  QUE TRATA DE LA CONDICIÓN")
        if inicio_idx == -1:
            inicio_idx = 0
            
        texto_final = texto[inicio_idx:]

        with open("input.txt", "w", encoding='utf-8') as f:
            f.write(texto_final)
        
        print(f"✅ ¡Conseguido! 'input.txt' creado con {len(texto_final)} caracteres.")

except Exception as e:
    print(f"❌ Error fatal: {e}")