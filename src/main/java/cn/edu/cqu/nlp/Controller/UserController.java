package cn.edu.cqu.nlp.Controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;

import cn.edu.cqu.nlp.Model.MyJson;
import cn.edu.cqu.nlp.Service.UserService;

@RestController
public class UserController {

	@Autowired
	UserService userService;

	// 提交并返回结果
	@PostMapping(value = "/commitText")
	@ResponseBody
	public MyJson commitText(String inputText) {
		Integer recordId = userService.commitText(inputText);
		userService.processModel(recordId);
		return userService.getResult(recordId);
	}

	// 历史结果
	@GetMapping(value = "/getRecords")
	@ResponseBody
	public MyJson getRecords() {
		return userService.getRecords();
	}
}
