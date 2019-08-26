#!/bin/bash
# sed -i 's/\r$//' run.sh
# remove data before running

# check parameter
# $1: exp_name
# $2: wait time
# $3: which gpu [0, 1, 2]
# $4: data_set [bike, taxi], both by default.

if [ ! $1 ];then
    echo "please input the experiment name."
    exit 0
fi
if [ ! $2 ];then
    echo "the wait time is seted to 1m by default"
fi
if [ ! $3 ];then
    echo "please input the gpu you want to use, check 'nvidia-smi' first."
    exit 0
fi
if [ ! $4 ]; then
    echo "dataset is set to both by default."
elif [[ "$4" != "taxi" ]] && [[ "$4" != "bike" ]]; then
    echo "can't recognize it. only bike and taxi."
    exit 0
fi

# delete old file: log in time
# if [ -f "./inter_data/result/${1}" ];then
#     rm -r inter_data/result/${1}
#    echo "remove inter_data/result/${1}"
# fi
rm_file(){
    for arg in "$@"
    do
        if [ -f $arg ]; then
            rm -r $arg
            echo "remove ./$arg"
        fi
    done
}
rm_file "log/${1}" "log/error_${1}" #"log/out_${1}"
echo "dataset is ${4:-both}"

params_name=$(python run.py --exp ${1} --type 1 --data ${4:-both} | sed 's/\[//g' | sed 's/\]//g' | sed 's/\,//g' | sed "s/'//g")
params_value=$(python run.py --exp ${1} --type 2 --data ${4:-both} | sed 's/\[//g' | sed 's/\]//g' | sed 's/\,//g' | sed "s/'//g")
params_length=$(python run.py --exp ${1} --type 3 --data ${4:-both})
value_length=$(python run.py --exp ${1} --type 4 --data ${4:-both})

echo "begin to run experiment: $1"
echo "parameter name: ${params_name}"
echo "parameter value combination: "
for combination in ${params_value}
do
    echo ${combination}
done
echo "the number of parameter: ${params_length}"
echo "the number of combination: ${value_length}"

command=""
cur_time=""
param_count=0
retry_count=0
params_name=(${params_name})
cur_number=1
success_number=0
pid=0
for each in ${params_value}
do
    for value in $(echo ${each} | sed 's/_/ /g')
    do
        command="${command} --${params_name[param_count]} ${value}"
        param_count=`expr ${param_count} + 1`
    done
    echo "--------------------------------------------------"
    echo "${cur_number}/${value_length}"
    
    echo "prepare to run: ${command}"
    
    nohup python ${1}.py --gpu ${3} ${command} > log/${1}_${4} 2>&1 &
    
    echo "current background job process is $(jobs -p), wait the process..."
    wait $(jobs -p)
    
    if [ $? -eq 0 ]; then
        cur_time=$(date "+%Y/%m/%d %H:%M:%S")
        echo "${cur_time}, command success"
        success_number=`expr ${success_number} + 1`
    else
        while [ ${retry_count} -lt 5 ]
        do
            cur_time=$(date "+%Y/%m/%d %H:%M:%S")
            retry_count=`expr ${retry_count} + 1`
            echo "${retry_count}/5  ${cur_time}, some error happen, wait ${2:-1m} to retry..."
            sleep ${2:-1m}
            nohup python ${1}.py --gpu ${3} ${command} > log/${1}_${4} 2>&1 &
            
            echo "current background job process is $(jobs -p), wait the process..."
            wait $(jobs -p)
            if [ $? -eq 0 ]; then
                break
            fi
        done
        
        # record the error
        if [ $retry_count -eq 5 ];then
            
            cur_time=$(date "+%Y/%m/%d %H:%M:%S")
            echo "command: nohup python ${1}.py ${command} > log/${1} 2>&1 &" >> log/error_${1}
            echo "time: ${cur_time}" >> log/error_${1}_${4}
            echo "--------------------------------------------------" >> log/error_${1}
            tail -n 10 log/${1}_${4} >> log/error_${1}_${4}
            echo "--------------------------------------------------" >> log/error_${1}
            
            echo "fail 5 times. this record has been writen into log/error_${1}"
        fi
    fi
    
    command=""
    param_count=0
    retry_count=0
    cur_number=`expr ${cur_number} + 1`
    
    echo "process finish, sleep ${2:-1m} then run next command..."
    sleep ${2:-1m}
done

echo "finish"
echo "total-success-fail ${value_length}-${success_number}-`expr ${value_length} - ${success_number}`"