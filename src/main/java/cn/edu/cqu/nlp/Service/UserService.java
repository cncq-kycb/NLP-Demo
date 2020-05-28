package cn.edu.cqu.nlp.Service;

import org.springframework.stereotype.Service;

import cn.edu.cqu.nlp.Model.MyJson;

@Service
public interface UserService {

	// 提交数据
	Integer commitText(String text);

	// 处理数据
	void processModel(Integer recordId);

	// 返回结果
	MyJson getResult(Integer recordId);

	// 获取历史结果
	MyJson getRecords();

}
