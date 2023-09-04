FROM nvidia/cuda:11.0.3-runtime-ubuntu20.04

WORKDIR /app

RUN apt-get update
# base
RUN apt-get install -y curl git
# python3
RUN apt-get install -y python3 python3-pip
# mecab
RUN apt-get install -y mecab libmecab-dev
#RUN apt-get install -y mecab-ipadic-utf8

# mecab-ipadic-NEologd
# root権限で動かすのでsudoは要らなさそうに見えるがインストーラーが要求してくるので必要
RUN apt-get install -y sudo
COPY ./mecab-ipadic-neologd ./mecab-ipadic-neologd
RUN ./mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -n -y

# タイムゾーン(ロギングのため)
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata

# 進捗表示
RUN apt-get install -y pv

# JSONのパース
RUN apt-get install -y jq

# shinra-classifyで必要
RUN apt-get install -y mecab-ipadic-utf8

# python3 packages
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt

# all the source
COPY . .

ENTRYPOINT ["./run.sh"]
