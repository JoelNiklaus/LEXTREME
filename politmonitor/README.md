# Politmonitor

Directory for the project POLITmonitor for the BFH

# Prepare everything

- For the code to work, you need to download the data from here: https://docs.fenceit.cloud/s/mmtfpn7WrSNa5NC. Just download there entire dataset by clicking on "Alle Daten herunterladen". Move the zip folder `Data.zip` to the folder `./data` (relative path).

- To use fasttext, go to https://fasttext.cc/docs/en/language-identification.html and download the model `lid.176.bin` and move it to the folder `models` (relative path).

- Finally, run `bash prepare.sh`.

# Important notes

- The excel file `topics_politmonitor.xlsx` lists the label `Grundkompetenzen_Illetrismus` (de), `Illetrisme` (fr), `Illetteratismo` (it) for topic id 10. This is actually outdated and must be `Europapolitik` (de) which is the label given in the excel file `topics_curia_vista.xlsx`.
