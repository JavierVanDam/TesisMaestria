from datetime import datetime
from requests_html import HTMLSession
import requests
import pandas as pd
import re
from time import sleep
import csv

PROJPATH = r"C:\Users\jvandam\PycharmProjects\DataScience"
archivoDescargas = 'LISTADO_DESCARGAS.csv'

dir(csv.reader)

csv.reader.__defaults__()


listadoYaCargado = []
with open(r'{}/{}'.format(PROJPATH,archivoDescargas), 'r') as archivo:
    reader = csv.reader(archivo, delimiter=";")
    for row in reader:
        listadoYaCargado.append(row[0])

# import pyppeteer.chromium_downloader as chiche
# chiche.chromium_executable()

df = pd.DataFrame(columns=["NOMBRE", "FECHA"]).astype(dtype={"NOMBRE": "object", "FECHA": "datetime64[ns]"})

se = HTMLSession()

url = 'https://www.metro951.com/deacaenmas/on-demand/#'

r = se.get(url)  # , params={'from':'10'}

sleep(5)

inicio = datetime.now()
r.html.render()
final = datetime.now()

print("TIEMPO RENDERIZADO: {}".format((final - inicio).total_seconds()))

cortes = r.html.xpath("//div[@class='cut']")

cortesDownloads = []
for c in cortes:
    cortesDownloads.append(c.xpath("//div[@class='cut__action' and starts-with(@onclick,'toDownload')]")[0])

hashNota = []
patronRE = re.compile(r"(toDownload\(\')(.+)(\'\)\;)")  # de esta regexp, me interesa el grupo 2

for c in cortesDownloads:
    # print(c.attrs['onclick'])
    hashNota.append(patronRE.search(c.attrs['onclick']).group(2))  # group(2) es el grupo q me interesa


hashNota = [h for h in hashNota if h not in listadoYaCargado]

excepciones = ['programa_completo', 'llamaatusabuelos', 'estrenos']
patronExcepciones = re.compile("|".join(excepciones))
hashNota = [h for h in hashNota if not (patronExcepciones.search(h))]

descargas = []
for h in hashNota:
    print("DESCARGANDO {}".format(h))
    req = requests.get('https://metro-on-demand.edge-apps.net/api/1.0/playlist/download?hash={}'.format(h))
    try:
        with open('{}.aac'.format(h), 'wb') as archivo:
            archivo.write(req.content)
            descargas.append({'NOMBRE': h, 'FECHA': datetime.now()})
    except Exception as e:
        print("ERROR CON {}".format(h))
        print(e)

df = df.append(pd.DataFrame(descargas))

df.to_csv(r'{}/{}'.format(PROJPATH,archivoDescargas), index=False, sep=";", mode='a', header=False)


r.close()
se.close()
