import torch

def main():
	print(torch.__version__)
	print(torch.cuda.is_available())

if __name__ == '__main__':
	main()
