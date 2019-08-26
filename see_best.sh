#!/bin/bash

result_path="inter_data/result"

# check if the file exist
if [ ! $1 ];then
    echo "you should give a name to me at least"
    exit 0
elif [ ! -f ${result_path}/$1 ];then
    echo "there is no such a file named $1 under ${result_path}"
    exit 0
else
    echo "fine file ${result_path}/$1"
fi
    
line_num=$(cat inter_data/result/$1 | wc -l)

# get the header of result file. In this form: 'A B C D E'
# method 1: data occupies two columns
# header=($(sed -n "1p" inter_data/result/$1 | sed 's/^[ \t]*//g' | sed 's/[ \t]*$//g' | sed 's/  \+/ /g'))
# column_num=$(expr ${#header[@]} + 1)
# method 2
column_num=$(sed -n "2p" $result_path/$1 | awk '{print NF}')

# check if the $2 is appropriate
if [ ! $2 ];then
    echo "choose the best 3 result by default"
elif [ $2 -gt `expr $line_num - 1` ];then
    echo "sorry, there is only `expr $line_num - 1` in the $1 file, please choose a number between [0, `expr $line_num - 1`]"
    exit 0
else
    echo "ok, prepare to analyse the bese $2 result for experiment $1"
fi

rmse_column=`expr $column_num - 1`

echo "the best ${2:-3} result for experiment $1"
echo "------------------------------------------------------------"
sed -n "1p" $result_path/$1
sed -n "2,${line_num}p" $result_path/$1 | sort -n -k ${rmse_column} | sed -n "1,${2:-3}p"