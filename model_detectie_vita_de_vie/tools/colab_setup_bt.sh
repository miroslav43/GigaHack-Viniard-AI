#!/bin/bash
# Downloads every dataset for the bunch+trunk model into a fresh Colab runtime, all in parallel.
# Each download writes /content/bt/dl_<name>.log, ending with "<NAME>_OK" on success and "END" when it stops.
# label map v8 is NOT here: it needs the Roboflow key, pasted by the user in a notebook cell (getpass).
mkdir -p /content/bt /content/tv/data/escayard/photos /content/tv/data/mildew
cd /content/bt

run() {  # run <name> <command...> in the background
  local name=$1; shift
  ( ( "$@" ) > "dl_$name.log" 2>&1; echo END >> "dl_$name.log" ) &
}

run pip bash -c 'pip -q install ultralytics==8.4.163 py7zr remotezip && echo PIP_OK'
run wgisd bash -c 'git clone -q --depth 1 https://github.com/thsant/wgisd.git /content/wgisd && echo WGISD_OK'
run pinheiro bash -c 'curl -sfL -o pinheiro.zip "https://zenodo.org/api/records/7717055/files/GrapevineBunchDetection.zip/content" \
  && unzip -q pinheiro.zip -d pinheiro && rm pinheiro.zip && echo PINHEIRO_OK'
run wgrape bash -c 'curl -sfL -o wgrape.zip "https://zenodo.org/api/records/4066730/files/wGrapeUNIPD-DL%20dataset.zip/content" \
  && unzip -q wgrape.zip -d wgrape && rm wgrape.zip && echo WGRAPE_OK'
run vairao bash -c 'curl -sfL -o /content/vairao.zip "https://zenodo.org/api/records/18152298/files/Dataset_Vairao_Grapevine_Trunks.zip/content" \
  && unzip -q /content/vairao.zip -d /content/vairao && rm /content/vairao.zip && echo VAIRAO_OK'
run humain bash -c 'curl -sfL -o humain.zip "https://github.com/humain-lab/vine-trunk/raw/master/Humain-Lab-vine-trunk-dataset.zip" \
  && unzip -q humain.zip -d /content/tv/data/humain && rm humain.zip && echo HUMAIN_OK'
run escayard bash -c 'curl -sfL -o /content/tv/data/escayard/datasheet.csv "https://zenodo.org/api/records/10362568/files/datasheet.csv/content" \
  && curl -sfL -o escayard.7z "https://zenodo.org/api/records/10362568/files/PLANTs_SmartPhonePhotos_JPG.7z/content" \
  && while ! grep -q PIP_OK dl_pip.log 2>/dev/null; do sleep 5; done \
  && python3 -c "import py7zr; py7zr.SevenZipFile(\"escayard.7z\").extractall(\"/content/tv/data/escayard/photos\")" \
  && rm escayard.7z && echo ESCAYARD_OK'
run mildew bash -c 'cd /content/tv/data/mildew && wget -q -U "Mozilla/5.0" -O m.zip \
  "https://data.mendeley.com/public-files/datasets/yxf2yv5ymt/files/baa9a436-bd95-425a-8d6f-aca7152cb1b0/file_downloaded" \
  && unzip -q m.zip && rm m.zip && echo MILDEW_OK'
run certh bash -c 'curl -sfL -o certh_ann.zip "https://zenodo.org/api/records/10777647/files/annotations.zip/content" \
  && unzip -q certh_ann.zip -d certh_ann \
  && while ! grep -q PIP_OK dl_pip.log 2>/dev/null; do sleep 5; done \
  && python3 /content/bt/certh_sub.py'
echo STARTED
