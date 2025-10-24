from astroquery.gaia import Gaia

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

print("Arquivo 'dados_orion.csv' salvo com sucesso!")