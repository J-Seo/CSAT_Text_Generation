# -*- coding: utf8 -*-

import plaidml.keras
import os

plaidml.keras.install_backend()

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras

import tensorflow as tf
import model as ml
import data
import numpy as np

import sys

from configs import DEFINES


# 구글 드라이브를 사용할 것이라면..? 여기다가 path 입력
DATA_OUT_PATH = './data_out/'

#def serving_input_receiver_fn():
#    receiver_tensor = {
#        'input': tf.placeholder(dtype=tf.int32, shape[None, DEFINES.max_sequence_length]),
#        'output': tf.placeholder(dtype=tf.int32, shape[None, DEFINES.max_sequence_length])
#    }
    
#    features = {
#        key: tensor for key, tensor in receiver_tensor.items()
#    }
#    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)

def main(self):
    data_out_path = os.path.join(os.getcwd(), DATA_OUT_PATH)
    os.makedirs(data_out_path, exist_ok=True)
    # 사전 데이터 불러오기 (인덱싱된 단어, 단어 자체, 단어 길이)
    char2idx, idx2char, vocabulary_length = data.load_vocabulary()
    # 훈련, 테스트용으로 input과 label을 가져온다
    train_input, train_label, eval_input, eval_label = data.load_data()

    # 훈련 인코딩 입력
    train_input_enc, train_input_enc_length = data.enc_processing(train_input, char2idx)
    # 훈련 디코딩 입력
    train_output_dec, train_output_dec_length = data.dec_output_processing(train_label, char2idx)
    # 훈련 디코딩 출력
    train_target_dec = data.dec_target_processing(train_label, char2idx)

    # 평가 인코딩 입력
    eval_input_enc, eval_input_enc_length = data.enc_processing(eval_input, char2idx)
    # 평가 디코딩 입력
    eval_output_dec, eval_output_dec_length = data.dec_output_processing(eval_label, char2idx)
    # 평가 디코딩 출력
    eval_target_dec = data.dec_target_processing(eval_label, char2idx)

    # 현재 경로'./'에 현재 경로 하부에 
    # 체크 포인트를 저장한 디렉토리를 설정
    check_point_path = os.path.join(os.getcwd(), DEFINES.check_point_path)
    # save_model_path = os.path.join(os.getcwd(), DEFINES.save_model_path)
    # 디렉토리를 만드는 함수이며 두번째 인자 exist_ok가 
    # True이면 디렉토리가 이미 존재해도 OSError가 
    # 발생하지 않는다.
    # exist_ok가 False이면 이미 존재하면 
    # OSError가 발생한다.
    os.makedirs(check_point_path, exist_ok = True)
    # os.makedirs(save_model_path, exist_ok = True)

    # 에스티메이터 구성한다.
    classifier = tf.estimator.Estimator(
        model_fn=ml.Model,  # 모델 등록한다.
        model_dir=DEFINES.check_point_path,  # 체크포인트 위치 등록한다.
        params={  # 모델 쪽으로 파라메터 전달한다.
            'embedding_size': DEFINES.embedding_size,
            'model_hidden_size': DEFINES.model_hidden_size,  # 가중치 크기 설정한다.
            'ffn_hidden_size': DEFINES.ffn_hidden_size,
            'attention_head_size': DEFINES.attention_head_size,
            'learning_rate': DEFINES.learning_rate,  # 학습율 설정한다.
            'vocabulary_length': vocabulary_length,  # 딕셔너리 크기를 설정한다.
            'embedding_size': DEFINES.embedding_size,  # 임베딩 크기를 설정한다.
            'layer_size': DEFINES.layer_size,
            'max_sequence_length': DEFINES.max_sequence_length,
            'xavier_initializer': DEFINES.xavier_initializer
        })

    # 학습 실행
    classifier.train(input_fn=lambda: data.train_input_fn(
        train_input_enc, train_output_dec, train_target_dec, DEFINES.batch_size), steps=DEFINES.train_steps)

    eval_result = classifier.evaluate(input_fn=lambda: data.eval_input_fn(
        eval_input_enc, eval_output_dec, eval_target_dec, DEFINES.batch_size))
    print('\nEVAL set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # 테스트용 데이터 만드는 부분이다.
    # 인코딩 부분 만든다.
    predic_input_enc, predic_input_enc_length = data.enc_processing(["Many people have pointed out that personal responsibility can often be lost in groups. Consider how often you’ve passed someone who is pulled over on the side of the road with car trouble. It is easy to justify your failure to help by telling yourself someone else will stop. Corporations add another layer of complication to the story of reduced personalresponsibility in group settings because corporations are set up to assign legal responsibility to the corporation itself instead of to its members. There are other complex organizations in which the individual members are often not sure of their impact or power to bring about change. In this kind of culture,“someone else will do it” is a common excuse."], char2idx)
    # 학습 과정이 아니므로 디코딩 입력은 
    # 존재하지 않는다.(구조를 맞추기 위해 넣는다.)
    predic_output_dec, predic_output_decLength = data.dec_output_processing([""], char2idx)
    # 학습 과정이 아니므로 디코딩 출력 부분도 
    # 존재하지 않는다.(구조를 맞추기 위해 넣는다.)
    predic_target_dec = data.dec_target_processing([""], char2idx)

    for i in range(DEFINES.max_sequence_length):
        if i > 0:
            predic_output_dec, predic_output_decLength = data.dec_output_processing([answer], char2idx)
            predic_target_dec = data.dec_target_processing([answer], char2idx)
        # 예측을 하는 부분이다.
        predictions = classifier.predict(
            input_fn=lambda: data.eval_input_fn(predic_input_enc, predic_output_dec, predic_target_dec, 1))

        answer, finished = data.pred_next_string(predictions, idx2char)

        if finished:
            break

    # 예측한 값을 인지 할 수 있도록
    # 텍스트로 변경하는 부분이다.
    print("answer: ", answer)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

tf.logging.set_verbosity
