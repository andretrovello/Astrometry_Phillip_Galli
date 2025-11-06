# Query Orion
'''from astroquery.gaia import Gaia

# A mesma query ADQL que você usou no site
query = """
SELECT
    source_id, ra, dec, parallax, parallax_error, pmra, pmdec,
    phot_g_mean_mag, bp_rp
FROM
    gaiadr3.gaia_source
WHERE
    1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', 83.83, -5.39, 1.0))
    AND parallax > 0
    AND parallax_over_error > 5
"""

# Executa a busca
job = Gaia.launch_job_async(query)
results = job.get_results()

# Salva os resultados diretamente em um arquivo CSV
results.write('dados_orion.csv', format='csv', overwrite=True)

print("Arquivo 'dados_orion.csv' salvo com sucesso!")'''

# Query Taurus

'''import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia

# 1. Definir as coordenadas centrais e o raio da busca
# Coordenadas ICRS: 04 41 01.0 +25 52 00 
coord_str = "04h41m01.0s +25d52m00s"
# Tamanho angular: 450 arcmin. Usamos metade disso para o raio.
radius_deg = 3.75  # (450 arcmin / 2) / 60 = 3.75 graus

# Converter as coordenadas para graus decimais
# O SkyCoord lida com a conversão de "hms dms" para "deg" automaticamente
c = SkyCoord(coord_str, frame='icrs')
ra_center = c.ra.degree
dec_center = c.dec.degree

print(f"Buscando em torno de RA: {ra_center:.4f} deg, Dec: {dec_center:.4f} deg")
print(f"Com um raio de: {radius_deg} graus")

# 2. Montar a query ADQL (linguagem de busca do Gaia)
# Vamos usar os mesmos filtros de qualidade que discutimos antes
query = f"""
SELECT
    source_id, ra, dec, parallax, parallax_error, pmra, pmdec,
    phot_g_mean_mag, bp_rp
FROM
    gaiadr3.gaia_source
WHERE
    1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra_center}, {dec_center}, {radius_deg}))
    AND parallax > 0
    AND parallax_over_error > 5
"""

''print("\nIniciando a busca no Gaia Archive...")
print("Isso pode levar alguns minutos, pois a região de Taurus é grande.")

# 3. Executar a busca de forma assíncrona
# (Isso é melhor para queries grandes, pois não trava seu computador)
job = Gaia.launch_job_async(query)
results = job.get_results()

print(f"Busca concluída! Foram encontradas {len(results)} fontes.")

# 4. Salvar os resultados em um arquivo CSV
output_filename = 'dados_taurus.csv'
results.write(output_filename, format='csv', overwrite=True)

print(f"Arquivo '{output_filename}' salvo com sucesso!")'''


# Query Phillip Galli et al. 2019
'''from astroquery.vizier import Vizier

# Identificador do catálogo no VizieR (baseado no bibcode J/Journal/Volume/Page)
catalog_id = 'J/A+A/630/A137'

print(f"Buscando tabelas para o catálogo: {catalog_id}")

# Instancia o Vizier
# Definimos row_limit=-1 para garantir que baixamos TODAS as linhas da tabela,
# e não apenas as 50 primeiras (que é o padrão).
v = Vizier(row_limit=-1)

# Baixa o catálogo. Isso retorna uma LISTA de tabelas.
# O 'result_tables' é um objeto especial do VizieR
result_tables = v.get_catalogs(catalog_id)

# Geralmente, a tabela principal que queremos é a primeira da lista.
# O artigo de Galli et al. tem duas tabelas. A Tabela 1 é a principal.
# Vamos acessar a primeira tabela (índice [0])
if result_tables:
    main_table = result_tables[0]
    print("\n--- Informações da Tabela Baixada ---")
    print(main_table.info)
    
    print(f"\nTotal de estrelas/objetos na tabela: {len(main_table)}")
    
    # Mostra as 5 primeiras linhas para confirmar
    print("\n--- 5 Primeiras Linhas da Tabela ---")
    print(main_table.to_pandas().head())

    # Agora você pode salvar em CSV se quiser
    main_table.write('galli_2019_table1.csv', format='csv', overwrite=True)
    print("\nTabela salva como 'galli_2019_table1.csv'")
else:
    print("Nenhuma tabela encontrada para este catálogo.")

    
'''




from astroquery.gaia import Gaia
import pandas as pd
import numpy as np
from astropy.table import Table

data = pd.read_csv('galli_2019_table1.csv')
main_table = pd.DataFrame(data)

# ---------------------------------------------------------------
# PASSO 1: CARREGAR SUA TABELA DO VIZIER
# ---------------------------------------------------------------
# Garanta que a variável 'main_table' exista.
# Se você estiver rodando o script do zero, você precisa
# carregar o arquivo CSV que você salvou do VizieR.
try:
    # Se você já tiver 'main_table' na memória (ex: num notebook), pule
    main_table
    print("Variável 'main_table' já existe na memória.")
except NameError:
    try:
        # Tente carregar do CSV
        # Substitua 'galli_2019_table1.csv' pelo nome que você usou
        main_table = Table.read('galli_2019_table1.csv') 
        print("Arquivo 'galli_2019_table1.csv' carregado com sucesso.")
    except FileNotFoundError:
        print("Erro: A variável 'main_table' não foi encontrada na memória.")
        print("E o arquivo 'galli_2019_table1.csv' não foi encontrado no disco.")
        print("Por favor, rode o script do VizieR primeiro ou verifique o nome do arquivo CSV.")
        raise

# ---------------------------------------------------------------
# PASSO 2: LIMPAR OS IDs
# ---------------------------------------------------------------
print("Iniciando a limpeza dos IDs...")
try:
    dr2_id_strings = main_table['GaiaDR2'].tolist()
    
    dr2_ids_cleaned_list = []
    ids_pulados = 0
    
    for s in dr2_id_strings:
        try:
            id_str = s.split(' ')[-1]
            numeric_id = np.int64(id_str) 
            dr2_ids_cleaned_list.append(numeric_id)
        except (AttributeError, IndexError, ValueError):
            ids_pulados += 1
            
    print(f"Encontrada e limpa a lista de {len(dr2_ids_cleaned_list)} IDs (DR2) válidos.")
    if ids_pulados > 0:
        print(f"({ids_pulados} linhas foram puladas por não terem um ID válido)")

except (KeyError) as e:
    print(f"Erro: A coluna 'GaiaDR2' não foi encontrada na 'main_table'.")
    raise

# ---------------------------------------------------------------
# PASSO 3: CRIAR A TABELA DE UPLOAD E A QUERY OTIMIZADA
# ---------------------------------------------------------------
upload_table = Table({'dr2_source_id_list': dr2_ids_cleaned_list})

# QUERY CORRIGIDA E OTIMIZADA:
# 1. Começa com a sua tabela (user_table)
# 2. Usa o nome correto da tabela (gaiadr3.dr2_neighbourhood)
query_dr3 = """
SELECT
    dr3.source_id, dr3.ra, dr3.dec, dr3.parallax, dr3.pmra, dr3.pmdec
FROM
    tap_upload.my_table AS user_table
JOIN
    gaiadr3.dr2_neighbourhood AS xmatch
    ON user_table.dr2_source_id_list = xmatch.dr2_source_id
JOIN
    gaiadr3.gaia_source AS dr3
    ON xmatch.dr3_source_id = dr3.source_id
"""

# ---------------------------------------------------------------
# PASSO 4: EXECUTAR A BUSCA
# ---------------------------------------------------------------
print("Iniciando a busca no Gaia DR3 (com query otimizada)...")

try:
    job = Gaia.launch_job_async(
        query=query_dr3,
        upload_resource=upload_table,
        upload_table_name="my_table",
        verbose=True  # Adiciona mais informações de debug
    )
    
    results_dr3_members = job.get_results()

    print(f"\nBusca concluída!")
    print(f"Foram encontrados dados no DR3 para {len(results_dr3_members)} das {len(dr2_ids_cleaned_list)} estrelas.")

    print("\n--- 5 primeiras linhas dos membros de Taurus (dados do DR3) ---")
    print(results_dr3_members.to_pandas().head())

    # Salve os resultados se desejar
    results_dr3_members.write('taurus_membros_dr3.csv', format='csv', overwrite=True)

except Exception as e:
    print(f"\nOcorreu um erro durante a busca no Gaia:")
    print(e)