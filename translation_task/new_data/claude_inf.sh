line_number=1
filename="$1"
folder="$2"
entity="$3"
while IFS= read -r line
do
	       if [ ! -f "${folder}/${folder}_test_${line_number}.txt" ]; then 
		aws bedrock-runtime invoke-model --model-id anthropic.claude-3-5-sonnet-20240620-v1:0 --body "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":
		    \"Please identity '${entity}' from the given text. Format the output as a list.\n\nText:\n$line\"}]}],\"anthropic_version\":\"bedrock-2023-05-31\",\"max_tokens\":2000,\"temperature\":0,\"top_k\":250,\"top_p\":0.999,\"stop_sequences\":[]}" --cli-binary-format raw-in-base64-out --region us-east-1 $folder/$folder_test_$line_number.txt
		printf "%d: %s\n" "$line_number" "$line"
		fi
		    ((line_number++))
	    done < "$filename"
