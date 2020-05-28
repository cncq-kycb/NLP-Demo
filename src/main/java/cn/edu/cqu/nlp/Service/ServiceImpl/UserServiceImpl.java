package cn.edu.cqu.nlp.Service.ServiceImpl;

import java.util.Date;
import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import cn.edu.cqu.nlp.Utils.Utils;
import cn.edu.cqu.nlp.Dao.RecordMapper;
import cn.edu.cqu.nlp.Model.MyJson;
import cn.edu.cqu.nlp.Model.RecordExample;
import cn.edu.cqu.nlp.Model.RecordWithBLOBs;
import cn.edu.cqu.nlp.Service.UserService;

@Component
public class UserServiceImpl implements UserService {

	@Autowired
	RecordMapper recordMapper;

	@Override
	public Integer commitText(String text) {
		Integer maxId = recordMapper.getId();
		RecordWithBLOBs recordWithBLOBs = new RecordWithBLOBs();
		recordWithBLOBs.setRecordId(maxId + 1);
		recordWithBLOBs.setInput(text);
		recordWithBLOBs.setRecordTime(new Date());
		try {
			recordMapper.insert(recordWithBLOBs);
			return new Integer(maxId + 1);
		} catch (Exception e) {
			System.err.println(e);
			return new Integer(-1);
		}
	}

	@Override
	public void processModel(Integer recordId) {
		String[] argv = new String[] { "/home/lys/anaconda3/envs/lys/bin/pythons",
				"./src/main/resources/script/lstm/predict.py", recordId + "" };
		String result;
		try {
			result = Utils.cmdCall(argv);
			System.out.println(result);
		} catch (Exception e) {
			System.err.println(e);
		}
//		RecordWithBLOBs record = new RecordWithBLOBs();
//		record.setRecordId(recordId);
//		record.setRecordTime(new Date());
//		record.setResult("0");
//		recordMapper.updateByPrimaryKeySelective(record);
	}

	@Override
	public MyJson getResult(Integer recordId) {
		try {
			RecordExample recordExample = new RecordExample();
			recordExample.or().andRecordIdEqualTo(recordId);
			List<RecordWithBLOBs> recordWithBLOBs = recordMapper.selectByExampleWithBLOBs(recordExample);
			if (recordWithBLOBs.size() != 0) {
				return new MyJson(recordWithBLOBs.get(0));
			}
		} catch (Exception e) {
			System.err.println(e);
		}
		return new MyJson(false, "查询出错");
	}

	@Override
	public MyJson getRecords() {
		try {
			RecordExample recordExample = new RecordExample();
			recordExample.or();
			recordExample.setOrderByClause("record_time DESC");
			List<RecordWithBLOBs> recordWithBLOBs = recordMapper.selectByExampleWithBLOBs(recordExample);
			if (recordWithBLOBs.size() != 0) {
				return new MyJson(recordWithBLOBs);
			}
		} catch (Exception e) {
			System.err.println(e);
		}
		return new MyJson(false, "查询出错");
	}
}
